# BINF6250 Project 9:

## Description of the project
This week's project was an implementation of the forward algorithm, backward algorithm, and posterior decoding. The forward algorithm and the backward algorithm both use dynamic programming to determine the probability of a sequence of emissions, given a Hidden Markov model. When their scoring matrices are used in tandem by the third algorithm, posterior decoding, the most probable state for each emission can be identified. 

## Dependencies
This week's code relies on our `core/HMModel_def.py` and `core/EmissionSet_and_HiddenState_defs.py` modules, making heavy use of the `HMModel` class and its associated methods to calculate probabilities from weights and to look up the relevant initial, emission, and state-transition probabilities. 

## Contents
This week's code is packaged in the corresponding jupyter notebook, `project09-forward-backward/project09.ipynb`. 

## Usage
The jupyter notebook containing this week's code is meant to be run from the `project09-forward-backward/` directory and handles importing our modules from the `core/` directory on its own as long as the two directories are under the same, shared parent directory. The algorithms can then be run individually from their associated cells in the Jupyter notebook, with a toy example able to be run across all three algorithms to demonstrate the differences in the forward and backward algorithms' scoring matrices. 

## Pseudocode

```
Forward Algo:
    Inputs:
        Sequence of emissions
        Model
    output:
        The entire scoring matrix (so we don't have to recalculate things when we want to look up positions later)
        The sum of the last column in the scoring matrix

    1. Make an M x N matrix where N is length of emissions sequence (assume it's a list of list of strings // list of strings)
       and M is the number of states in our model
    2. Compute column 0 using: (remember to convert these probabilities to log space)
        initial probabilities for each hidden state
        Emission probabilities at t=0
    3. Iterate through the emissions in the sequence (enumerate here to access corresponding column number in scoring matrix)
            Skip or ignore the first emission
        4. Iterate through each hidden state
            5. We need to calculate Emission probability in this square P(observation at current t | current hidden state)
                Convert this to logspace
            6. Initialize a sum at 0
            7. Iterate through all of the hidden states at the previous timepoint (t-1)
                8. Look up transition probability from prev_state (@ t-1) to curr_state (@ t), and convert to log
                9. Add the score from this cell (previous state at t) plus the log-probability of the transition plus the current (@ t) emission log-prob
                10.np.logaddexp() this to the sum from step 6 (if we weren't in log space, the log-probabilities above would be raw probabilities that we'd multiply, and then in this step we would just add them to the sum)

            11.Now we have the score for this position in the matrix, write it in
    12.Add up the values in the last column, this is the probability of the sequence given the model
    13.Return the scoring matrix and the probability of the sequence given the model


Backward Algo:
    Inputs:
        Sequence of emissions
        Model
    output:
        The entire scoring matrix (so we don't have to recalculate things when we want to look up positions later)
        The sum of the last column in the scoring matrix

    1. Make an M x N matrix where N is length of emissions sequence (assume it's a list of list of strings // list of strings)
       and M is the number of states in our model
    2. Initialize column N with all 1s
    3. Iterate through the emissions in the sequence from back to front (enumerate here to access corresponding column number in scoring matrix)
            Skip or ignore the last emission
        4. Iterate through each hidden state
            5. Initialize a sum at 0
            6. Iterate through all of the hidden states at the next timepoint (t+1)
                7. Look up transition probability from curr_state (@ t) to next_state (@ t+1), and convert to log
                8. Add the following:
                    score from the cell we're looking forward to (state at t+1)
                    log-probability of the transition plus the current (@ t) emission log-prob
                    log of emission probability for emission at t+1 given the state we're looking forward to at t+1
                9. np.logaddexp() this to the sum from step 6 (if we weren't in log space, the log-probabilities above would be raw probabilities that we'd multiply, and then in this step we would just add them to the sum)

            10.Now we have the score for this position in the matrix, write it in
    10.5. Make a beta column and compute the probabilities of the paths going to the beta column using initial probs instead of transition probs (so do all the above stuff for it)
    11.Add up the values in the first column, this is the probability of the sequence given the model
    12.Return the scoring matrix and the probability of the sequence given the model


Helper func to combine forward and backward probability for current position:
    Takes as input:
        Forwards scoring matrix
        Backwards scoring matrix
        Position (row and column number), representing combination of an emission at time t and being in a given state at that time
        Probability of the sequence given the model (could pass either forward or backwards probs here)
    1. Look up the values in both matrices for the given position
    2. Apply the formula:
        foward_prob(current_position) * backward_prob(current_position) / prob(entire sequence | the model)
        Note that our lookups from the forward and backwards matrices are in logspace, so actually just do forward_score + backward_score - log(prob(entire seq | model))
    3. Return the result of that formula


Posterior-decoding:
    Takes as input:
        Sequence of emissions
        The model
    0. Make an ordered list of the states
    1. Make the forwards matrix and the backwards matrix
    2. Initialize a traceback list
    3. Iterate through emissions in the emissions sequence
        4. Start an empty list for forward-backward probabilities
        5. Insert the forward-backward probability for all states into the list (the indices of this prob-list correlate 1:1 with the indices of the list of states)
        6. Grab the argmax (the index of the state with the highest probability at this emission) from the list of fb-probabilities and add it to the traceback
    7. Could convvert traceback list from being a list of indices to a list of state names for readability
        [state[idx] for idx in traceback] boom easy comprehension <3
    8. Return the list of most likely states

```

## Personal reflections
### Successes

### Strugges
While we were planning out the backward algorithm, we found our resources quite ambiguous as to whether we needed to account for the initial probabilities once we reached the beginning of the sequence. We ended up reasoning that, since the forward and backward algo's are expected to reach the same probability and the forward algorithm includes the initial probabilities when it initializes its scoring matrix, then the backward algorithm would need to account for the intial probabilities at the end. This is why we chose to add the initial probabilities at the start instead of just adding up the values in the first column of the scoring matrix. Our resources from class were also a little ambiguous about which emission's probability gets calculated for the backwards probability at any given position. Since the emission at position i is accounted for in the forward probability of a given state at position i, we reasoned that it wouldn't make sense to double count it. This led to us looking ahead to all the probabilities of the emission at position i+1 under all the states we could go to from where we are when calculating the backwards probability for the state we're looking at in position i. 

### Individual reflections

#### Hongyuan Deng: 
we implemented the Forward, Backward, and Posterior Decoding algorithms for Hidden Markov Models (HMM) from scratch. Transitioning from an agricultural background, tackling dynamic programming and Object-Oriented Programming (OOP) was a steep learning curve. However, we successfully conquered the math behind log-space transformations to prevent numerical underflow and mastered vectorized computing using NumPy tensors. Seeing our code accurately decode the hidden states of genomic sequences was incredibly rewarding.

#### Justin:
This week's project was much more relaxed than last week's, primarily due to last week having demanded us to plan and write our HMM module in addition to implementing Viterbi. The forward algorithm, backward algorithm, and posterior decoding are relatively straightforward, and we were able to inform a lot of how we went about implementing those algorithms using our understanding of the dynamic programming from the Viterbi algorithm. The forward algorithm, especially, took little time for us to plan out, since the only change from Viterbi was adding the probabilities instead of chooding the greatest one. As tempting as it was to copy-paste our pseudocode from project 8 to speed up our planning session, I wanted to write out the pseudocode from scratch this week to avoid any oversights we might give rise to by only spot-checking for changes we needed to make to the Viterbi pseudocode. Our planning session went really well, and we were able to bang out the pseudocode for all three of the algorithms in a couple hours, which was so much faster than the three days we took to write our HMM module last week. We all felt pretty confident in how we wanted to implement the three algorithms this week, and we were able to reason through any ambiguities left open in the lecture. 

#### Linh
I think we didn't really have any issues with the algorithm this week. After doing the pseudocode together, everyone just sort of did their own implementation. I liked that I was able to take advantage of all the parallels between viterbi and this week's algorithms (forward especially) as not only did that make it simple to "write" forward through essentially editing viterbi, it actually was somewhat enlightening in terms of probability theory and model assumptions. But I also understand why others would rather treat the algorithms separately, and the different approaches allow us to explore possibilities and see the strengths of methods that might not come naturally to oneself.

I highly recommend anyone struggling with understanding these sorts of probability-based algorithms but aren't particularly interested in math to still take the hit and freshen up on notation, then seek out a few online sources that explain the same thing in addition to the lecture slides, since a lot of them are ambiguous about some aspects and/or explain concepts in ways that do not match how you intuit. Seeking out multiple sources also mitigates the damage from random rendering errors, rare variable name conventions, different fonts for greek letters etc. causing confusion. 

## Appendix
Generative AI was not used on this project. 
