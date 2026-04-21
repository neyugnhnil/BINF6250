from core.HMModel_def import HMModel
from mathhelpers import logp

def setup_HMModel_lookups(emissions: list, model: HMModel, log_toggle: bool = True):

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
    value_to_index = [{value_name: j for j, value_name in enumerate(es.value_names)} for es in emission_sets]
    coded_emissions = []
    for obs in emissions:
        coded_obs = []
        for i, value in enumerate(obs):
            coded_obs.append(value_to_index[i][value])
        coded_emissions.append(coded_obs)

    # choose probability conversion function
    convert_prob = logp if log_toggle else (lambda p: p)

    # init_probs[hs index] = init prob for that hs
    init_probs = [convert_prob(model.P_init[state_name]) for state_name in states]

    # trans_probs[previous state index][current state index] = trans prob for those hs
    trans_probs = [
        [convert_prob(model.P_hh[prev_state][curr_state]) for curr_state in states]
        for prev_state in states
        ]

    # emit_hs_probs[es index][hs index][value index] = prob of that emission-set value given that hs
    emit_hs_probs = []
    for x, es in enumerate(emission_sets):
        set_name = es.set_name
        emit_hs_probs.append(
            [
                [convert_prob(p) for p in model.P_eh[set_name][state_name]]
                for state_name in states
            ]
        )

    return {
        "states": states,
        "emission_sets": emission_sets,
        "n_states": n_states,
        "n_sets": n_sets,
        "n": n,
        "coded_emissions": coded_emissions,
        "value_to_index": value_to_index,
        "init_probs": init_probs,
        "trans_probs": trans_probs,
        "emit_hs_probs": emit_hs_probs,
        "log_toggle": log_toggle,
    }