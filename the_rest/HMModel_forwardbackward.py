from core.HMModel_def import HMModel
from .setup_HMModel_lookups import setup_HMModel_lookups
import math
import numpy as np

# helper to sum more than two logs at once
def logsumexp(log_values: list[float]) -> float:
    return float(np.logaddexp.reduce(log_values))


def forward_distribution(emissions: list, model: HMModel):
    #### SETUP AND VALIDATION ####
    lookup = setup_HMModel_lookups(emissions, model, log_toggle=True)

    n_states = lookup["n_states"]
    n_sets = lookup["n_sets"]
    n = lookup["n"]
    coded_emissions = lookup["coded_emissions"]
    init_logprobs = lookup["init_probs"]
    trans_logprobs = lookup["trans_probs"]
    emit_logtables = lookup["emit_hs_probs"]

    # forward_table[hs index][position index] = logp(e_0...e_i, s_i = hs)
    forward_table = [[float("-inf")] * n for _ in range(n_states)]
    prev_scores = [float("-inf")] * n_states

    #### FORWARD ALGORITHM ####
    # at each position, we rebuild prev_scores and move forward one position

    # P(s_0, e_0) = P_init(s) x P(emission|s)
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
        forward_table[s][0] = prev_scores[s]

    # when i > 0:
    # P(s_i) = sum for each s_(i-1) : P(s_i-1) x P(s_i-1 -> s_i) x P(e_i|s_i)
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
            incoming_scores = []

            eprob = curr_emit[curr_s]

            for prev_s in range(n_states):
                score = prev_scores[prev_s] + trans_logprobs[prev_s][curr_s] + eprob
                incoming_scores.append(score)

            curr_scores[curr_s] = logsumexp(incoming_scores)
            forward_table[curr_s][i] = curr_scores[curr_s]

        prev_scores = curr_scores

    # get the sequence total logprob
    total_logprob = logsumexp(prev_scores)

    return forward_table, total_logprob

def backward_distribution(emissions: list, model: HMModel):
    #### SETUP AND VALIDATION ####
    lookup = setup_HMModel_lookups(emissions, model, log_toggle=True)

    n_states = lookup["n_states"]
    n_sets = lookup["n_sets"]
    n = lookup["n"]
    coded_emissions = lookup["coded_emissions"]
    init_logprobs = lookup["init_probs"]
    trans_logprobs = lookup["trans_probs"]
    emit_logtables = lookup["emit_hs_probs"]

    # backward_table[hs index][position index] = logp(e_i+1...e_n-1 | s_i = hs)
    backward_table = [[float("-inf")] * n for _ in range(n_states)]
    next_scores = [float("-inf")] * n_states

    #### BACKWARD ALGORITHM####
    # at each position, we rebuild next_scores and move backward one position

    # at the last position there are no future emissions
    for s in range(n_states):
        next_scores[s] = 0.0                                               # log(1) = 0
        backward_table[s][n - 1] = next_scores[s]

    # when i < n-1:
    # P(s_i) = sum for each s_i+1 : P(s_i+1) x P(s_i -> s_i+1) x P(e_i+1|s_i+1)
    for i in range(n - 2, -1, -1):                                         # for position i...
        next_obs = coded_emissions[i + 1]                                  # these are the next observations

        # precompute P(next emission|next s) for the NEXT position
        next_emit = [float("-inf")] * n_states
        for s in range(n_states):
            log_emit = 0.0                                                 # log(1) = 0
            for x in range(n_sets):
                obs_index = next_obs[x]                                    # get observation for each emission set
                log_emit += emit_logtables[x][s][obs_index]
            next_emit[s] = log_emit

        curr_scores = [float("-inf")] * n_states
        for curr_s in range(n_states):
            outgoing_scores = []

            for next_s in range(n_states):
                score = trans_logprobs[curr_s][next_s] + next_emit[next_s] + next_scores[next_s]
                outgoing_scores.append(score)

            curr_scores[curr_s] = logsumexp(outgoing_scores)
            backward_table[curr_s][i] = curr_scores[curr_s]

        next_scores = curr_scores

    # get start scores for each hs 
    first_obs = coded_emissions[0]
    start_scores = []
    for s in range(n_states):
        log_emit = 0.0                                                     # log(1) = 0
        for x in range(n_sets):
            obs_index = first_obs[x]
            log_emit += emit_logtables[x][s][obs_index]

        start_scores.append(init_logprobs[s] + log_emit + backward_table[s][0])

    # total sequence logprob
    total_logprob = logsumexp(start_scores)

    return backward_table, total_logprob

def _valid_distribution_table(table, n_states: int, n: int):
    if not isinstance(table, list):
        return False
    if len(table) != n_states:
        return False
    for row in table:
        if not isinstance(row, list):
            return False
        if len(row) != n:
            return False
    return True

def posterior_decoding(emissions: list, model: HMModel, forward_table = None, backward_table = None) -> list[str]:
    #### SETUP AND VALIDATION ####
    lookup = setup_HMModel_lookups(emissions, model, log_toggle=True)
    states = lookup["states"]
    n_states = lookup["n_states"]
    n = lookup["n"]

    #### GET DISTRIBUTIONS ####
    # if the distributions weren't provided make them on the spot

    # forward
    if not _valid_distribution_table(forward_table, n_states, n):
        forward_table, total_logprob = forward_distribution(emissions, model)
    else:
        # total sequence prob comes from the forward distribution here
        last_column = [forward_table[s][n - 1] for s in range(n_states)]
        total_logprob = logsumexp(last_column)
    
    # backwards
    if not _valid_distribution_table(backward_table, n_states, n):
        backward_table, _ = backward_distribution(emissions, model)

    #### POSTERIOR DECODING ####
    # at each position, choose the hs with maximal posterior probability
    path = []

    for i in range(n):
        best_state = None
        best_score = float("-inf")
        for s in range(n_states):
            posterior_logprob = forward_table[s][i] + backward_table[s][i] - total_logprob
            if posterior_logprob > best_score:
                best_score = posterior_logprob
                best_state = s
        path.append(states[best_state])

    return path

# testing
# from .fakedata_for_HMModel import emissions, model
# from .HMModel_viterbi import viterbi

# fdist = forward_distribution(emissions, model)

# #print(fdist[0])
# print('fdist total sequence logprob=',fdist[1])
# bdist = backward_distribution(emissions, model)

# #print(bdist[0])
# print('\nbdist total sequence logprob=',bdist[1])

# fbpath = posterior_decoding(emissions, model, fdist, bdist)

# print('\nposterior decoding path=',fbpath)

# viterbipath = viterbi(emissions, model)

# print('\nviterbi path=', viterbipath)