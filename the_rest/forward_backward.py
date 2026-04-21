# Justin's script for the forward, backward, and forward-backward algorithms
# NOTE: the imports in this script assume it's being run in the root directory of the repo

# imports
import sys
import numpy as np
from pathlib import Path

"""# set up to import from another directory in this project
repo_root = Path.cwd().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))"""

# import from our own modules
from core.HMModel_def import HMModel
from core.EmissionSet_and_HiddenState_defs import EmissionSet, HiddenState
from the_rest.viterbi import identify_emissionset


def forward(model: HMModel, emission_seq: list[str], verbose: bool = False) -> tuple[float, np.ndarray]:
    """
    Function to determine the probabiltiy of a sequence of emissions from front to back, given a hidden Markov model
    also generates the scoring matrix needed for forward-backward

    Parameters
    ----------
    model : HMModel
        the hidden markov model whose initial, state transition, and emission probabilities
        are used to determine the optimal sequence of hidden states
    emission_seq : list[str]
        list of emissions, where each emission is a string of unknown length
    verbose : bool, optional
        determines if debugging print statements get printed, by default False

    Returns
    -------
    tuple[float, np.ndarray]
        float: the log-probability of the emission_seq, given the model
        np.ndarray: the scoring matrix under the foward algorithm
    """
    if verbose:
        print("====Starting forward algorithm====")

    # attempt to get the name for the emission set we should query for this seq of emissions
    curr_emission_set = identify_emissionset(emission_seq, model, verbose=verbose)
    if curr_emission_set is None:
        raise ValueError("This sequence of emissions contains a combination of emissions "
                         f"that is not used by any emission sets in {model}")
    
    # convert weights in the model to probabilities
    model.normalize_all()

    # make a list of the states in the model
    states = [state.hidden_state_name for state in model.hidden_states]
    
    # make a scoring and traceback matrix for dynamic programming
    scores = np.full((len(states), len(emission_seq)), -np.inf)

    # for the first emission, we need to use the initial probs instead of transition probs
    curr_emission = emission_seq[0]
    # get the index for this emission in the current state's emission probabilities
    emission_idx = curr_emission_set.value_names.index(curr_emission)

    # iterate over the states
    for state_idx, curr_state in enumerate(states):
        # the model's initial prob's table is indexed the same way as the list of states
        scores[state_idx, 0] = np.log(model.P_init[curr_state]) + np.log(model.P_eh[curr_emission_set.set_name][curr_state][emission_idx])  # type: ignore
    
    # now we handle every single emission beyond the first
    for i, curr_emission in enumerate(emission_seq):
        if i == 0:
            # we've already filled out column zero on our scores and traceback matrix
            continue
        # get the index for this emission in the current state's emission probabilities
        emission_idx = curr_emission_set.value_names.index(curr_emission)
        
        # iterate through the states we could be at this timepoint
        for state_idx, curr_state in enumerate(states):
            # initialize the running sum of scores in this position coming from every direction
            # and log(x) -> -inf as x -> 0
            sum_score = -np.inf

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
                # if verbose:
                #    print(f"At coords: ({state_idx},{i})")
                #    print(f"prev_score: {prev_score}, transition_prob: {transition_prob}, log(trans):{np.log(transition_prob)}, "
                #          f"emission_prob:{emit_prob}, log(emit_prob):{np.log(emit_prob)}, leads to current score:{curr_score}")

                # add to the running sum the probability of going to the current state from this previous state
                # since we're in logspace, just logaddexp it to simulate the effect of doing log(prob1 + prob2)
                if verbose:
                    print(f"Running probability sum (sum_score): {sum_score}, adding curr_score: {curr_score} to it")
                sum_score = np.logaddexp(sum_score, curr_score)
                if verbose:
                    print(f"Now sum_score updated to: {sum_score}")
            
            # write the score in for this position (the log of the sum of probabilities of all previous states leading
            # to the current one)
            scores[state_idx, i] = sum_score
    
    # add the values in the last column of the matrix to get P(sequence | model)
    last_col_score = np.logaddexp.reduce(scores[:, len(emission_seq) - 1])
    if verbose:
        print(f"Last column: {scores[:, len(emission_seq) - 1]}, "
              f"with probabilities: {[np.exp(score) for score in scores[:, len(emission_seq)-1]]}")
    # break out of logspace to report as a real probability
    # seq_prob = np.exp(last_col_score)

    return last_col_score, scores


def _backwards_fill_column(scoring_matrix: np.ndarray, 
                           col_num: int,
                           emission_idx: int,
                           model: HMModel, 
                           states: list[str], 
                           curr_emission_set: EmissionSet,
                           verbose: bool = False) -> None:
    """
    Function to update a given column (at col_num) of the scoring matrix
    for the backwards algorithm (so it looks ahead to column col_num+1
    to calculate the scores for col_num)

    Parameters
    ----------
    scoring_matrix: np.ndarray
        The scoring matrix being updated, assumed to be an existing 2darray
    col_num: int
        The column number being updated (CANNOT be the last column, at index width-1)
    emission_idx: int
        The index of the emission in the emission set being referneced to look up probabilities in model
    model: HMModel
        The hidden markov model used for probabilty lookups
    states: list[str]
        List of state names, used for probability lookup
    curr_emission_set: EmissionSet
        The EmissionSet being used to look up probabilities in the model
    verbose: bool, false by default
        Prints debug statements if true

    Returns
    -------
    Nothing, just updates the matrix passed as a parameter
    """
    # iterate through the states we could be at this timepoint
    for state_idx, curr_state in enumerate(states):
        # initialize the running sum of scores in this position coming from every direction
        # and log(x) -> -inf as x -> 0
        sum_score = -np.inf

        # iterate through the states the PREVIOUS EMISSION could have been on
        for next_idx, next_state in enumerate(states):
            # grab the next state's score
            next_score = scoring_matrix[next_idx, col_num+1]
            # get the probability of transition from the current state to the next one (don't invert directionality here)
            transition_prob = model.P_hh[curr_state][next_state]  # type: ignore
            # get the next position's emission probability
            emit_prob = model.P_eh[curr_emission_set.set_name][next_state][emission_idx]  # type: ignore
            # get the current state's score contribution to the next state at position i+1
            curr_score = next_score + np.log(transition_prob) + np.log(emit_prob)
            # if verbose:
            #    print(f"At coords: ({state_idx},{col_num})")
            #    print(f"next_score: {next_score}, transition_prob: {transition_prob}, log(trans):{np.log(transition_prob)}, "
            #            f"emission_prob:{emit_prob}, log(emit_prob):{np.log(emit_prob)}, leads to current score:{curr_score}")

            # add to the running sum the probability of going to the next state from this state
            # since we're in logspace, just logaddexp it to simulate the effect of doing log(prob1 + prob2)
            if verbose:
                print(f"Running probability sum (sum_score): {sum_score}, adding to it curr_score: {curr_score}")
            sum_score = np.logaddexp(sum_score, curr_score)
            if verbose:
                print(f"Now sum_score updated to: {sum_score}")
        
        # write the score in for this position (the log of the sum of probabilities of all previous states leading
        # to the current one)
        scoring_matrix[state_idx, col_num] = sum_score
        


def backward(model: HMModel, emission_seq: list[str], verbose: bool = False) -> tuple[float, np.ndarray]:
    """
    Function to determine the probabiltiy of a sequence of emissions from back to front, given a hidden Markov model
    also generates the scoring matrix needed for forward-backward

    Parameters
    ----------
    model : HMModel
        the hidden markov model whose initial, state transition, and emission probabilities
        are used to determine the optimal sequence of hidden states
    emission_seq : list[str]
        list of emissions, where each emission is a string of unknown length
    verbose : bool, optional
        determines if debugging print statements get printed, by default False

    Returns
    -------
    tuple[float, np.ndarray]
        float: the log-probability of the emission_seq, given the model
        np.ndarray: the scoring matrix under the backward algorithm (has one more column than 
        that of the forward algo due to needing to compute a column for the starting position)
    """
    if verbose:
        print("====Starting backward algorithm====")
    
    # attempt to get the name for the emission set we should query for this seq of emissions
    curr_emission_set = identify_emissionset(emission_seq, model, verbose=verbose)
    if curr_emission_set is None:
        raise ValueError("This sequence of emissions contains a combination of emissions "
                         f"that is not used by any emission sets in {model}")
    
    # convert weights in the model to probabilities
    model.normalize_all()

    # make a list of the states in the model
    states = [state.hidden_state_name for state in model.hidden_states]
    
    # make a scoring and traceback matrix for dynamic programming
    scores = np.full((len(states), len(emission_seq)), -np.inf)

    # iterate over the states to update the last column
    for state_idx, curr_state in enumerate(states):
        # here, we ignore the last emission (we'll look at it at column N-1) and act as though
        # we're starting with initial probabilities of 1
        scores[state_idx, -1] = np.log(1)
    
    # now we handle every single column from the second-to-last to the first
    for i in range(len(emission_seq)-2, 0, -1):
        # get the index for the next position's emission
        emission_idx = curr_emission_set.value_names.index(emission_seq[i+1])
        
        # use a helper func to fill in this column
        _backwards_fill_column(scores, i, emission_idx, model, states, curr_emission_set, verbose)
    
    # the for loop ignored the first (i=0) column
    emission_idx = curr_emission_set.value_names.index(emission_seq[1])
    _backwards_fill_column(scores, 0, emission_idx, model, states, curr_emission_set, verbose)

    # and we haven't accounted for our initial prob's yet, so start a score for a shared starting position
    init_score = -np.inf
    # grab the index for the emission at t=0
    emission_idx = curr_emission_set.value_names.index(emission_seq[0])
    # iterate through all the states we can go to from the initial position (beta)
    for next_idx, next_state in enumerate(states):
        # get the initial probability for the next state
        init_logprob = np.log(model.P_init[next_state])  # type: ignore
        # get the next position's emission probability
        emit_prob = model.P_eh[curr_emission_set.set_name][next_state][emission_idx]  # type: ignore
         # grab the next state's score at t=0
        next_score = scores[next_idx, 0]
        
        # calculate the score for the path going into this position from the initial state
        curr_score = init_logprob + np.log(emit_prob) + next_score

        # logaddexp this score to the log of the running sum of probabilities
        init_score = np.logaddexp(init_score, curr_score)

    # break out of logspace to report the probability as a probability
    # seq_prob = np.exp(init_score)
    
    # now append the beta column to the front of the scoring matrix
    beta_column = np.full(shape=(len(states), 1), fill_value=init_score)
    scores = np.hstack([beta_column, scores])

    return init_score, scores


def _forward_backward(forward_matrix: np.ndarray, 
                      backward_matrix: np.ndarray, 
                      position: tuple[int, int], 
                      seq_logprob: float) -> float:
    """
    Function to apply the formula for the forward-backward algo
    NOTE, IMPORTANT: this formula uses log-probabilities
    """
    # look up the values for the given position in both matrices
    row = position[0]
    col = position[1]
    forward_logprob = forward_matrix[row, col]
    # the lookup in the backward matrix needs to account for
    # the extra column for the initial state
    backward_logprob = backward_matrix[row, col + 1]

    # apply the formula: fw_prob * bk_prob / prob(entire_seq)
    # except we're in logspace
    return forward_logprob + backward_logprob - seq_logprob


def posterior_decoding(model: HMModel, emission_seq: list[str], verbose: bool = False) -> list[str]:
    """
    Function to apply the forward and backward algorithms to determine the
    most likely state at a given position in the emissions sequence

    Parameters
    ----------
    model : HMModel
        model being used for probability lookups
    emission_seq : list[str]
        sequence of emissions being analyzed
    verbose : bool, optional
        prints debug messages if true, by default False
    
    Returns
    -------
    list[str]
        Sequence representing the most probable state at each position
        Has same shape as emission_seq
        Uses the names of the states in model
    """
    # make an ordered list of the states
    states = [state.hidden_state_name for state in model.hidden_states]

    # make the forwards and backwards matrix
    forward_logprob, forward_matrix = forward(model, emission_seq, verbose)
    backward_logprob, backward_matrix = backward(model, emissions, verbose)
    seq_logprob = (forward_logprob + backward_logprob) / 2

    if verbose:
        print("Starting posterior-decoding:\n"
          f"Forward algo output:\nLog-probability of sequence under the model: {round(forward_logprob, 15)}"
          f"\n{forward_matrix}"
          f"Backward algo output:\nLog-probability of sequence under the model: {round(backward_logprob, 15)}"
          f"\n{backward_matrix}")

    # initialize list of optimal states at each time
    probable_states = []

    # iterate through the emissions in the emissions sequence
    for col in range(len(emission_seq)):
        # make a list of the forward-backward probability for all states
        fb_logprobs = [_forward_backward(forward_matrix, backward_matrix, (row, col), seq_logprob) for row in range(len(states))]
        
        if verbose:
            print(f"Forward-backward log-probs at column {col}: {fb_logprobs}")

        # grab the argmax of the forward-backward log-probabilities
        # the state with the greatest log-prob is also the state with the greatest prob
        # no need to leave logspace to make this comparison
        best_idx = np.argmax(fb_logprobs)
        probable_states.append(states[best_idx])
    
    return probable_states


if __name__ == "__main__":
    # generate some toy data:
    # emission sets
    dna_base = EmissionSet(
        name="dna_base",
        length=4,
        value_names=["A", "T", "C", "G"],
        default_weights=[1, 1, 1, 1]
        )

    methylation = EmissionSet(
        name="methylation",
        length=2,
        value_names=["unmethylated", "methylated"],
        default_weights=[2, 1]
        )

    # hidden states
    background = HiddenState(
        name="background",
        init_weight=3.0,
        emission_weights={
            "dna_base": [2, 2, 1, 1],
            "methylation": [9, 1]
        }
    )

    CpG_island = HiddenState(
        name="CpG_island",
        init_weight=1.0,
        emission_weights={
            "dna_base": [1, 1, 5, 5],
            "methylation": [1, 9]
        }
    )

    # build model
    model = HMModel(
        emission_sets=[dna_base, methylation],
        hidden_states=[background, CpG_island]
        )

    # specify transition weights
    model.replace_transition_row("background", [9, 1])
    model.replace_transition_row("CpG_island", [1, 9])

    # fake observations
    emissions = list("ATCGCATCCCTCGC")
    print(f"Sequence of emissions is:\n{emissions}")

    # do forward-backward on the seq
    probable_states = posterior_decoding(model, emissions, verbose=False)
    print(f"Posterior-decoding output:\n{probable_states}")
