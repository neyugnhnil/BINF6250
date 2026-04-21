import math
import numpy as np

###### WIP ###############################

from .nthHMModel_def import nthHMModel

def nthviterbi(emissions: list, model: nthHMModel) -> list[str]:
    
    #### SETUP AND VALIDATION ####
    model.normalize_all() # make sure probabilities exist

    states = [hs.hidden_state_name for hs in model.hidden_states]
    emission_sets = model.emission_sets
    n_states = len(states)
    n_sets = len(emission_sets)
    n = len(emissions)
    order = model.markov_order

    if n < order:
        raise Exception(f"sequence length must be at least markov_order ({order})")

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

    # safe log probability helper
    def logp(p: float) -> float:
        if p <= 0:
            return float("-inf")
        return math.log(p)

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

    # init_logprobs[history indices] = init log prob for that starting history
    init_logprobs = np.full(model.P_init.shape, float("-inf"), dtype=float)
    init_mask = model.P_init > 0
    init_logprobs[init_mask] = np.log(model.P_init[init_mask])
    
    # trans_logprobs[history indices][current state index] = trans log prob for those hs
    trans_logprobs = np.full(model.P_trans.shape, float("-inf"), dtype=float)
    trans_mask = model.P_trans > 0
    trans_logprobs[trans_mask] = np.log(model.P_trans[trans_mask])

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
    # traceback[position][current history] = previous history that gave the best score
    traceback = [dict() for _ in range(n)]

    # we are only ever concerned with the previous "layer" of scores
    # score histories are tuples of hidden-state indices of length = order
    prev_scores = {}

    #### VITERBI ####
    # conceptually: at each position, we rebuild prev_scores and move forward one position

    # P(s_0,...,s_order-1) = P_init(history) x product of P(emission|s_t) for the states in that history
    # we compute the probabilities for each possible starting history
    for history in np.ndindex(init_logprobs.shape):
        base_score = init_logprobs[history]

        if base_score == float("-inf"):
            continue

        log_emit = 0.0
        for pos, s in enumerate(history):
            obs = coded_emissions[pos]

            # assuming conditional independence,
            # we can add the log probs of obs from each set for a total "emission log prob"
            for x in range(n_sets):
                obs_index = obs[x]                                          # get observation for each emission set
                log_emit += emit_logtables[x][s][obs_index]

        prev_scores[history] = base_score + log_emit
        traceback[order - 1][history] = None

    # when i >= order:
    # best score ending in history_i = best previous path score + transition log prob + emission log prob
    # for each candidate current state, we vary the previous history and keep the maximal score
    # we also keep track of what previous history provided that maximum score (traceback)

    for i in range(order, n):                                               # for position i...
        obs = coded_emissions[i]                                            # these are the observations

        # precompute P(emission|s) for this position
        curr_emit = [float("-inf")] * n_states
        for s in range(n_states):
            log_emit = 0.0                                                  # log(1) = 0
            for x in range(n_sets):
                obs_index = obs[x]                                          # get observation for each emission set
                log_emit += emit_logtables[x][s][obs_index]
            curr_emit[s] = log_emit

        curr_scores = {}
        for history, prev_score in prev_scores.items():
            for curr_s in range(n_states):
                best_score = prev_score + trans_logprobs[history + (curr_s,)] + curr_emit[curr_s]
                # P(history_i|history_i-1) = P(history_i-1) x P_transition(history_i-1,curr_s) x P(emission|curr_s)

                new_history = history[1:] + (curr_s,)

                if (new_history not in curr_scores) or (best_score > curr_scores[new_history]):
                    curr_scores[new_history] = best_score
                    traceback[i][new_history] = history

        prev_scores = curr_scores

    # termination
    best_last = max(prev_scores, key=lambda history: prev_scores[history])  # what's the best ending history?

    #### TRACEBACK ####
    history_path = [best_last]

    for i in range(n - 1, order - 1, -1):
        history_path.append(traceback[i][history_path[-1]]) # what previous history allowed the best score for this history?

    history_path.reverse()

    # first history contributes all its states, each later history contributes only its last state
    path = list(history_path[0])
    for history in history_path[1:]:
        path.append(history[-1])
    
    return [states[s] for s in path]