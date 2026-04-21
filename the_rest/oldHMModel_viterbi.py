from core.HMModel_def import HMModel
import math

# safe log probability helper
def logp(p: float) -> float:
    if p <= 0:
        return float("-inf")
    return math.log(p)

# viterbi function assuming 1st order markov model
def viterbi(emissions: list, model: HMModel) -> list[str]:
    
    #### SETUP AND VALIDATION ####
    model.normalize_all() # make sure probabilities exist

    states = [hs.hidden_state_name for hs in model.hidden_states]
    emission_sets = model.emission_sets
    n_states = len(states)
    n_sets = len(emission_sets)
    n = len(emissions)

    # standardize emissions shape
    if n_sets == 1: 
        # accept list, convert list to list of lists
        emissions = [[obs] for obs in emissions]
    else:
        # require each observation to be a list of strings of length n_sets
        for obs in emissions: 
            if not isinstance(obs, list):
                raise Exception("for multiple emission sets, each observation must be a list matching model.emission_sets order")
            if len(obs) != n_sets:
                raise Exception(f"each observation must have length {n_sets} ")

    #### PREPPING LOOKUP OBJECTS ####

    # all observations can be represented as a list of indices 
    # ie. if coded_emissions[0] = [1,2], then the first observation had value 1 for the first es, value 2 for the second
    value_to_index = [{value_name: j for j, value_name in enumerate(es.value_names)} for es in emission_sets]
    # (it's better to make the map for each emission set than to scan with .index() in the next loop)
    coded_emissions = []
    for obs in emissions:
        coded_obs = []
        for i, value in enumerate(obs):
            coded_obs.append(value_to_index[i][value])
        coded_emissions.append(coded_obs)

    # init_logprobs[hs index] = init log prob for that hs
    init_logprobs = [logp(model.P_init[state_name]) for state_name in states]
    
    # trans_logprobs[previous state index][current state index] = trans log prob for those hs
    trans_logprobs = [
        [logp(model.P_hh[prev_state][curr_state]) for curr_state in states]
        for prev_state in states
        ]

    # emit_logtables[es index][hs index][value index] = log prob of that emission-set value given that hs
    emit_logtables = []
    for x, es in enumerate(emission_sets):
        set_name = es.set_name
        emit_logtables.append(
            [
                [logp(p) for p in model.P_eh[set_name][state_name]]
                for state_name in states
            ]
        )

    # empty traceback lookup
    traceback = [[None] * n for _ in range(n_states)]

    # we are only ever concerned with the previous "column" of scores
    # empty container for that:
    prev_scores = [float("-inf")] * n_states

    #### VITERBI ####
    # conceptually: at each position, we rebuild prev_scores and move forward one position

    # P(s_0) = P_init(s) x P(emission|s)
    # we compute the probabilities for each s_0
    first_obs = coded_emissions[0]
    for s in range(n_states):
        log_emit = 0.0                                                      # log(1) = 0

        # assuming conditional independence, 
        # we can add the log probs of obs from each set for a total "emission log prob"
        for x in range(n_sets):
            obs_index = first_obs[x]                                        # get observation for each emission set
            log_emit += emit_logtables[x][s][obs_index]
        
        prev_scores[s] = init_logprobs[s] + log_emit

    # when i > 0:
    # best score ending in s_i = best previous path score + transition log prob + emission log prob
    # for each s_i, we vary s_i-1 and keep the maximal score
    # we also keep track of what s_i-1 provided that maximum score (traceback)

    for i in range(1, n):                                                   # for position i...
        obs = coded_emissions[i]                                            # these are the observations

        # precompute P(emission|s) for this position
        curr_emit = [float("-inf")] * n_states
        for s in range(n_states):
            log_emit = 0.0                                                  # log(1) = 0
            for x in range(n_sets):
                obs_index = obs[x]                                          # get observation for each emission set
                log_emit += emit_logtables[x][s][obs_index]
            curr_emit[s] = log_emit                                         

        curr_scores = [float("-inf")] * n_states
        for curr_s in range(n_states):
            best_score = float("-inf")
            best_prev = None

            eprob = curr_emit[curr_s]

            for prev_s in range(n_states):
                score = prev_scores[prev_s] + trans_logprobs[prev_s][curr_s] + eprob
                # P(s_i|s_i-1) = P(s-1) x P_transition(s_i-1,s_i) x P(emission|s_i)

                if score > best_score:
                    best_score = score
                    best_prev = prev_s

            curr_scores[curr_s] = best_score
            traceback[curr_s][i] = best_prev

        prev_scores = curr_scores

    # termination
    best_last = max(range(n_states), key=lambda s: prev_scores[s])  # what's the "row number" of the best prev_score?

    #### TRACEBACK ####
    path = [best_last]

    for i in range(n - 1, 0, -1):
        path.append(traceback[path[-1]][i]) # what previous state allowed the best score for this state?
    
    path.reverse()
    
    return [states[s] for s in path]


## test ##
from .fakedata_for_HMModel import emissions, model

path = viterbi(emissions, model)

print(path)