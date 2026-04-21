from core.HMModel_def import HMModel
from .setup_HMModel_lookups import setup_HMModel_lookups
import math

# viterbi function assuming 1st order markov model
def viterbi(emissions: list, model: HMModel) -> list[str]:
    
    #### SETUP AND VALIDATION ####
    lookup = setup_HMModel_lookups(emissions, model, log_toggle=True)

    states = lookup["states"]
    n_states = lookup["n_states"]
    n_sets = lookup["n_sets"]
    n = lookup["n"]
    coded_emissions = lookup["coded_emissions"]
    init_logprobs = lookup["init_probs"]
    trans_logprobs = lookup["trans_probs"]
    emit_logtables = lookup["emit_hs_probs"]

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
# from .fakedata_for_HMModel import emissions, model

# path = viterbi(emissions, model)

# print(path)