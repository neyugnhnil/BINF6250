import numpy as np

from typing import Optional

from core.HMModel_def import HMModel, EmissionSet, HiddenState
from core.mathhelpers import logp_np, logsumexp


##### HMModel <-> array conversion helpers #####
# instead of rebuilding base no-numpy HMModel

# helperhelper to enforce using one emission set only 
def _resolve_emission_set_name(model: HMModel, which_emission_set: Optional[str]) -> str:
    # resolve which emission set should be used for baum-welch in case of multiple emissions
    set_names = [es.set_name for es in model.emission_sets]

    if which_emission_set is None:
        if len(set_names) == 1:
            return set_names[0]
        raise Exception("model contains multiple emission sets. which_emission_set must be provided.")
    if which_emission_set not in set_names:
        raise Exception(f"emission set {which_emission_set} not found in model")

    return which_emission_set

# helper to extract parameter arrays, ie. P_init, P_hh, P_eh, ie. pi, A, B 
def _extract_parameter_arrays(model: HMModel, set_name: str):
    # extract normalized parameters from a HMModel in array form
    # convert our model's weights to probabilities
    model.normalize_all()

    # make a list of hidden state names
    states = [hs.hidden_state_name for hs in model.hidden_states]

    # convert the initial probability dict into an np array
    P_init = np.array(
        [model.P_init[state_name] for state_name in states],
        dtype=float
        ) # ie. pi, shape (N,)

    # convert the state transition dict into a 2d numpy array
    P_hh = np.array(
        [[model.P_hh[prev_state][curr_state] for curr_state in states] for prev_state in states],
        dtype=float
        ) # ie. A, shape (N, N)

    # convert the emission dict into a 2d numpy array 
    # (also reduce the dimensionality caused by having multiple emission sets in the emissions dict)
    P_eh = np.array(
        [model.P_eh[set_name][state_name] for state_name in states],
        dtype=float
        ) # ie. B, shape (N, K)

    return P_init, P_hh, P_eh

# helper that writes new parameters to a HMModel
# to be called at the end of parameter estimation
def _write_probability_arrays_to_model(model: HMModel, set_name: str, P_init: np.ndarray, P_hh: np.ndarray, P_eh: np.ndarray):

    # update init weights
    for i, hidden_state in enumerate(model.hidden_states):
        hidden_state.init_weight = float(P_init[i])

    # update transition weights
    model.W_hh = P_hh.tolist() 
    # our "matrix" in base HMModel is a list of lists

    # update emission weights (for chosen emission set)
    for s, hidden_state in enumerate(model.hidden_states):
        hidden_state.emission_weights[set_name] = P_eh[s].tolist()

    model.clear_derived()
    model.normalize_all()

##### forward / backward in log space #####
# so we don't have to leave log space every iteration for no reason
def _forward_log(logp_init: np.ndarray, logp_hh: np.ndarray, emit_logp: np.ndarray):
    # takes advantage of the fact that emit_logp at each t is fixed for all hidden states
    # T represents the timepoints (uppercase T = the end), N is number of states
    T, N = emit_logp.shape
    # initialize a scoring matrix for the forward algo
    log_forward_dist = np.full((T, N), -np.inf, dtype=float) 
    log_forward_dist[0] = logp_init + emit_logp[0]

    for t in range(1, T):
        scores = log_forward_dist[t - 1][:, None] + logp_hh
        log_forward_dist[t] = emit_logp[t] + logsumexp(scores, axis=0)

    logp_seq = float(logsumexp(log_forward_dist[T - 1], axis=0))
    return log_forward_dist, logp_seq

def _backward_log(logp_hh: np.ndarray, emit_logp: np.ndarray):
    # takes advantage of the fact that emit_logp at each t is fixed for all h
    T, N = emit_logp.shape
    log_backward_dist = np.full((T, N), -np.inf, dtype=float)
    log_backward_dist[T - 1] = 0.0

    for t in range(T - 2, -1, -1):
        scores = (logp_hh + emit_logp[t + 1][None, :] + log_backward_dist[t + 1][None, :])
        log_backward_dist[t] = logsumexp(scores, axis=1)

    return log_backward_dist

##### ACTUAL ALGORITHM ####

def baumwelch(
    model: HMModel,                                 # HMModel with starting parameters and emission schema
    emissions: list,                                # observed sequence for the chosen emission set
    which_emission_set: Optional[str] = None,       # which emission set to use; None allowed if only one in HMModel
    burn_in: int = 0,                               # number of iterations before convergence is checked
    convergence_check_frequency: int = 1,           # check convergence every x iterations after burn-in
    tol: float = 1E-5,                              # convergence threshold on absolute log-likelihood change
    max_iter: int = 500,                            # maximum number of iterations
    pseudocount: float = 1E-7,                      # additive regularization strength for init / transition / emission expected counts
    return_model: bool = False                      # whether to return updated HMModel or just arrays/diagnostics 
    ):
 
    #### VALIDATION ####
    if len(model.emission_sets) == 0:
        raise Exception("model must contain at least one emission set")
    if len(model.hidden_states) == 0:
        raise Exception("model must contain at least one hidden state")
    if len(emissions) == 0:
        raise Exception("emissions must contain at least one observation")
    if burn_in < 0:
        raise Exception("burn_in must be >= 0")
    if convergence_check_frequency <= 0:
        raise Exception("convergence_check_frequency must be >= 1")
    if max_iter <= 0:
        raise Exception("max_iter must be >= 1")
    if pseudocount < 0:
        raise Exception("pseudocount must be >= 0")
    # etc 

    #### RESOLVE EMISSION SET NAME ####
    set_name = _resolve_emission_set_name(model, which_emission_set)
    emission_set = model.get_es(set_name)

    #### ENCODE EMISSION SEQUENCE AS INDICES ####
    value_to_index = {value_name: i for i, value_name in enumerate(emission_set.value_names)}
    coded_emissions = []
    for obs in emissions:
        if obs not in value_to_index:
            raise Exception(f"observation {obs} not found in emission set {set_name}")
        coded_emissions.append(value_to_index[obs])
    coded_emissions = np.array(coded_emissions, dtype=int)                  # shape (T,); the sequence

    #### INITIALIZE FROM MODEL ####
    # turn raw P_* params into arrays
    P_init, P_hh, P_eh = _extract_parameter_arrays(model, set_name)
    
    # take probabilities to log space 
    logp_init = logp_np(P_init)                                             # shape (N,)
    logp_hh = logp_np(P_hh)                                                 # shape (N, N)
    logp_eh = logp_np(P_eh)                                                 # shape (N, K)

    # get constant scalars 
    T = len(coded_emissions)                # timesteps
    N = len(model.hidden_states)            # hidden states
    K = P_eh.shape[1]                       # emission values

    # take pseudocount to log space
    log_pseudocount = float("-inf") if pseudocount == 0 else np.log(float(pseudocount))

    # prepare to MAXIMIZE EXPECTATION
    logp_seq_history = [] # mostly for diagnostics and maybe plotting convergence
    last_checked_logp_seq = None
    converged = False

    #### EM ITERATIONS ####
    for iteration in range(max_iter):

        #### REBUILD PER-TIMESTEP EMISSION LOG-PROBS ####

        # emit_logp[t, s] = log(P(observation_t | hidden_state_s))
        # for all rows, take column at t1, then column at t2, then column at t3...
        emit_logp = logp_eh[:, coded_emissions].T                           # shape (T, N)
        # (T,N) is more convenient than (N,T) for forward-backward 

        #### E-STEP: COMPUTE LIKELIHOODS ####

        # get log forward distribution and logp_seq
        log_forward_dist, logp_seq = _forward_log(logp_init, logp_hh, emit_logp)

        # get log backward distribution
        log_backward_dist = _backward_log(logp_hh, emit_logp)

        # for diagnostic output / maybe plotting convergence
        logp_seq_history.append(float(logp_seq))

        # log_gamma[t, i] = log(forward[t][i]) + log(backwards[t][i]) - logp_seq
        log_gamma = log_forward_dist + log_backward_dist - logp_seq                         # shape (T, N)

        # log_ksi[t, i, j] =                        # overall shape (T-1, N, N) = (time, source hs, next hs)
        log_ksi = (
            log_forward_dist[:-1, :, None]          # log(forward[t][i])                    shape (T-1, N, 1)
            + logp_hh[None, :, :]                   # + log(P(i->j))                        shape (1, N, N)
            + emit_logp[1:, None, :]                # + log(P(next emission|next state))    shape (T-1, 1, N)
            + log_backward_dist[1:, None, :]        # + log(backward[t+1][j])               shape (T-1, 1, N)
            - logp_seq                              # - logp_seq                            scalar
            )                                                           
        # numpy broadcasting: two dims are compatible if they are equal or one of them is 1

        #### CONVERGENCE CHECK ####
        checkcon1 = (iteration >= burn_in)
        checkcon2 = ((iteration - burn_in) % convergence_check_frequency == 0)
        if checkcon1 and checkcon2:
            if last_checked_logp_seq is not None and abs(logp_seq - last_checked_logp_seq) < tol:
                converged = True
                break
            # else, update logp
            last_checked_logp_seq = float(logp_seq)

        #### M-STEP: RE-ESTIMATE PARAMETERS ####

        #### initial distribution update ####

        # log expected count of each state at t=0
        log_init_counts = log_gamma[0]                              
        
        if pseudocount > 0:
            # add pseudocount to every state's expected count at t=0
            log_init_counts = np.logaddexp(log_init_counts, log_pseudocount)
            
            # the denom is the sum of numerators 
            # so pseudocount has already been added
        
        # normalize
        logp_init = log_init_counts - logsumexp(log_init_counts, axis=0)

        #### transition distribution update ####

        # numerator: log_hh_num[i, j] is log expected count of transition i->j
        log_hh_num = logsumexp(log_ksi, axis=0)                     # shape (N, N)

        # denom: log_hh_den[i] is log expected count of times i is the source state regardless of destination 
        log_hh_den = logsumexp(log_gamma[:-1], axis=0)              # shape (N,)
        
        if pseudocount > 0:
            # add pseudocount to each transition's expected count 
            log_hh_num = np.logaddexp(log_hh_num, log_pseudocount)
            
            # the denom must then increase by (pseudocount * number of possible destinations)
            log_hh_den = np.logaddexp(log_hh_den, np.log(N) + log_pseudocount)
        
        # normalize row-wise (ie. by source state)
        logp_hh = log_hh_num - log_hh_den[:, None]                  # shape (N, N)

        #### emission distribution update ####

        # numerator: log_eh_num[i, k] is expected count of being in state i and emitting k 
        # numerator step 1: prepare one-hot masks (does the same thing as indicator function)
        log_one_hot = np.full((T, K), -np.inf)
        log_one_hot[np.arange(T), coded_emissions] = 0.0                                    # shape (T, K)   
        # at row/time t, value is log(0)=neginf for unobserved emissions, log(1)=0 for observed                   

        # numerator step 2: 
        # (T,N,1) + (T,1,K) -> (time, state, don't vary) + (time, don't vary, emission)
        # axis=0 -> sum over dimension T
        log_eh_num = logsumexp(log_gamma[:, :, None] + log_one_hot[:, None, :], axis=0)     # shape (N, K)
        # ie: sum_t(gamma_t(i) but only if we observed k)
        # so log_eh_num[i, k] is an expected-given-i tally of k accross T
    
        # denominator: log_eh_den[i] is log expected count of the state being i at all
        log_eh_den = logsumexp(log_gamma, axis=0)                                           # shape (N,)

        if pseudocount > 0:
            # add pseudocount to each (i, k) combo's expected count
            log_eh_num = np.logaddexp(log_eh_num, log_pseudocount)

            # the denom must then increase by (pseudocount * number of possible emissions)
            log_eh_den = np.logaddexp(log_eh_den, np.log(K) + log_pseudocount)

        # normalize row-wise (ie. by hidden state)
        logp_eh = log_eh_num - log_eh_den[:, None]                                         # shape (N, K)

    #### WRITE FINAL PARAMETERS BACK TO MODEL ####
    P_init = np.exp(logp_init)
    P_hh = np.exp(logp_hh)
    P_eh = np.exp(logp_eh)

    _write_probability_arrays_to_model(model, set_name, P_init, P_hh, P_eh)

    if return_model:
        return model

    return {
        "which_emission_set": set_name,
        "coded_emissions": coded_emissions,
        "P_init": P_init,
        "P_hh": P_hh,
        "P_eh": P_eh,
        "logp_init": logp_init,
        "logp_hh": logp_hh,
        "logp_eh": logp_eh,
        "logp_seq_history": logp_seq_history,
        "n_iterations": len(logp_seq_history),
        "converged": converged,
        "pseudocount": pseudocount
        }

#test

# emit = EmissionSet(
#     name = "energy", 
#     length = 2, 
#     value_names=["tired", "not tired"], 
#     default_weights=[1.0, 1.0]
#     )

# state1 = HiddenState(
#     name="slept well",
#     init_weight=1.0,
#     emission_weights={"energy": [2.0, 1.0]}
#     )

# state2 = HiddenState(
#     name="slept poorly",
#     init_weight=1.0,
#     emission_weights={"energy": [4.0, 1.0]}
#     )

# model = HMModel(emission_sets=[emit], hidden_states=[state1, state2])
# model.W_hh = [[1.0, 1.0],[1.0, 1.0]]

# emissions = ["tired", "tired", "tired", "not tired", "tired", 
#                 "not tired", "not tired", "not tired",
#                 "tired", "tired", "tired", "not tired", "tired", "not tired", 
#                 "tired", "tired", "tired", "tired", "tired", "tired",
#                 "tired", "tired", "tired", "tired", "tired", "tired", "not tired", 
#                 "tired", "not tired", "tired"]

# result = baumwelch(model=model,emissions=emissions,max_iter=100,pseudocount=1e-6,return_model=False)

# print("P_init:")
# print(result["P_init"])
# print()

# print("P_hh:")
# print(result["P_hh"])
# print()

# print("P_eh:")
# print(result["P_eh"])
# print()

# print("logp_seq_history:")
# print(result["logp_seq_history"])
# print()

# print("converged:")
# print(result["converged"])
