# Justin, Linh, Hongyuan
# Project 8: Viterbi Algorithm

# import statements
import numpy as np
from core.EmissionSet_and_HiddenState_defs import EmissionSet, HiddenState
from core.HMModel_def import HMModel


def identify_emissionset(emission_seq: list[str], model: HMModel, verbose: bool = False) -> EmissionSet | None:
    """
    Function to identify the name for the emission set that should be queried
    when looking up the probabilities in the given model for the supplied 
    sequence of emissions

    Parameters
    ----------
    emission_seq : list[str]
        the sequence of emissions we're checking against the model
    model: HMModel
        The hidden markov model whose emission sets are being queried here

    Returns
    -------
    str
        the name of the emission set that should be used to look up 
        probabilities for this sequence of emissions
    returns None if no match is found
    """
    # get the unique emissions in the sequence
    unique_emissions = set(emission_seq)
    if verbose:
        print(f"Unique emissions in sequence: {unique_emissions}")

    # iterate over the model's emission sets
    for curr_emission_set in model.emission_sets:
        # convert the list of unique emission names to a set
        legal_emissions = set(curr_emission_set.value_names)
        if verbose:
            print(f"Current EmissionSet: {curr_emission_set.set_name}\n"
                f"Legal emissions: {legal_emissions}")

        # see what emissions are in the sequence that aren't in the current emission set
        diff = unique_emissions - legal_emissions
        if verbose:
            print(f"Elements in emission seq that aren't legal: {diff}")
        
        # if they're the same, return the name of the emission set
        if not diff:
            return curr_emission_set
    
    # if we make it out here, there's no match so return none
    return None


def viterbi(emission_seq: list[str], model: HMModel, verbose: bool = False) -> list[str]:
    """
    Function implementing the Viterbi algorithm to determine the optimal
    sequence of hidden states that could have given rise to the sequence
    of emissions in emission_seq

    Parameters
    ----------
    emission_seq : list[str]
        list of emissions, where each emission is a string of unknown length
    model : HMModel
        the hidden markov model whose initial, state transition, and emission probabilities
        are used to determine the optimal sequence of hidden states

    Returns
    -------
    list[str]
        sequence equal in length to the emission_seq, containing the hidden state at each emission
    """
    # attempt to get the name for the emission set we should query for this seq of emissions
    curr_emission_set = identify_emissionset(emission_seq, model, verbose=verbose)
    if curr_emission_set is None:
        print(f"This sequence of emissions contains a combination of emissions that is not used by any emission sets in {model}")
        return []

    # convert weights in the model to probabilities
    model.normalize_all()

    # make a list of the states in the model
    states = [state.hidden_state_name for state in model.hidden_states]

    # make a scoring and traceback matrix for dynamic programming
    scores = np.full((len(states), len(emission_seq)), -np.inf)
    traceback = np.zeros((len(states), len(emission_seq)), dtype=np.int64)

    # for the first emission, we need to use the initial probs instead of transition probs
    curr_emission = emission_seq[0]
    # get the index for this emission in the current state's emission probabilities
    emission_idx = curr_emission_set.value_names.index(curr_emission)

    # iterate over the states
    for state_idx, curr_state in enumerate(states):
        # the model's initial prob's table is indexed the same way as the list of states
        scores[state_idx, 0] = np.log(model.P_init[curr_state]) + np.log(model.P_eh[curr_emission_set.set_name][curr_state][emission_idx])  # type: ignore
        traceback[state_idx, 0] = state_idx  # set to a temp value for now
    # print(f"at t=0, scores matrix looks like:\n{scores}\n and traceback matrix looks like:\n{traceback}")
    # now we handle every single emission beyond the first
    for i, curr_emission in enumerate(emission_seq):
        # Previously, this iterated over enumerate(emission_seq[1:]), but that caused issues due to i starting at 0 anyway
        if i == 0:
            # we've already filled out column zero on our scores and traceback matrix
            continue
        # get the index for this emission in the current state's emission probabilities
        emission_idx = curr_emission_set.value_names.index(curr_emission)
        
        # iterate through the states we could be at this timepoint
        for state_idx, curr_state in enumerate(states):
            # initialize the current best score and current best previous score
            best_score = -np.inf
            best_prev = 40
            # get the current emission probability
            emit_prob = model.P_eh[curr_emission_set.set_name][curr_state][emission_idx]  # type: ignore

            # iterate through the states the PREVIOUS EMISSION could have been on
            for prev_idx, prev_state in enumerate(states):
                # grab the previous state's score
                prev_score = scores[prev_idx, i-1]
                # print(f"Accessing score at {prev_idx}, {i-1} = {prev_score}")
                transition_prob = model.P_hh[prev_state][curr_state]  # type: ignore
                # get the current state's score if its traceback is from prev_state
                curr_score = prev_score + np.log(transition_prob) + np.log(emit_prob)

                # check if this candidate score is greater than best_score and update accordingly
                # print(f"At coords: ({state_idx},{i}) Comparing current score:{curr_score} to running best score {best_score}")
                if curr_score > best_score:
                    # print(f"Updating high score for i={i} at state={state_idx} to {curr_score} from state {prev_idx}")
                    best_score = curr_score
                    best_prev = prev_idx
            
            # write the best score and its traceback to the matrices
            scores[state_idx, i] = best_score
            traceback[state_idx, i] = best_prev

    # pick the starting point for the traceback (the highest score at the last column/timepoint)
    curr_state_idx = np.argmax(scores[:, len(emission_seq) - 1])
    # get the state at the last timepoint and add its name to the path
    state_at_t = states[curr_state_idx]
    path = [state_at_t]
    # print(f"Before looping, highest scored state at t={len(emission_seq) - 1} is {state_at_t}. Path is {path}")

    # iterate through the columns of the traceback matrix (will skip t=0)
    # this looks backwards by one timepoint to see where the current position got
    # its score from and then adds that previous state that to the path (we don't miss the last state
    # because we already added the state at t=end to the path)
    # this avoids index errors from looking backwards while t=0
    # print(emission_seq)
    # print(f"Emission sequence is of length: {len(emission_seq)}. Attempting to access t={len(emission_seq) - 1}")
    for t in range(len(emission_seq) - 1, 0, -1):

        # get the state at time t-1 that gives the current state at t its score
        where_from = traceback[curr_state_idx, t]
        # print(f"At t={t}, {states[curr_state_idx]} gets its score from state {where_from} ({states[where_from]})")
        # add this state to the path
        path.append(states[where_from])
        # print(f"At t={t}, path looks like {path}")
        # move to the state at t-1 before t ticks down
        curr_state_idx = where_from

    if verbose:
        print(f"Score matrix:{scores.shape}\n{scores}")
        print(f"Traceback matrix:{traceback.shape}\n{traceback}")

    # raise an error if our path is somehow the wrong length (points to an issue during traceback)
    if len(path) != len(emission_seq):
        raise Exception(f"The length of the path, {len(path)}, is not equal to the length of the sequence of emissions, {len(emission_seq)}")
    
    # since the traceback was done from end to start, reverse the path
    path.reverse()

    # remember when we put a tmep value in at column 0 of the traceback? It was because the init_state wasn't in
    # the list of states we wanted to iterate through when making our dynamic programming matrices. However, regardless of 
    # which state we're in at t=0, that position ALWAYS gets its score frm the initial state
    path[0] = "initial_state"

    return path



if __name__ == "__main__":
    from .fakedata_for_HMModel import emissions, model

    # our toy data includes multiple different emissions for each timepoint
    # I just want to access the nucleotide emissions
    simplified_emissions = [emission[0] for emission in emissions]
    print(f"Output:\n{viterbi(simplified_emissions, model, verbose=True)}")