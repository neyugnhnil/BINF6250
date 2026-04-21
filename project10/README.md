# BINF6250 Project 10:

## Description of the project
This week's project was an implementation of the Baum-Welch algorithm for training a Hidden Markov Model with a pre-determined architecture off of a sequence of emissions with unknown states. This marked the third week of working with the HMM module we wrote on week 8, directly using the algorithms we've worked on in the past two weeks to train a model, rather than computing an optimal path or a sequence's probabilty as we did previously. We had found that the calculations for Baum-Welch would be made easier if our HMM's data structures were expressed as arrays, rather than as dictionaries and lists as they we had originally developed them. To avoid refactoring our code for the HMM module, Viterbi, or the Forward and Backward algorithms, we made a helper function that runs inside of Baum-Welch to convert our HMM's datastructures into numpy arrays, which allowed us to more easily perform summations and elementwise operations for Baum-Welch without needing to iterate over keys. 

## Dependencies
This project's notebook is dependent on the HMM module we had developed in `/core` to import the HMModel object and the `mathhelpers` module we wrote with helper functions for common operations we made use of in the past three weeks' projects.  
We use pathlib and sys to manage the imports in `/project10/project10.ipynb` from the sister directory's modules `/core/HMModel_def` and `/core/mathhelpers`. 

## Contents
We have placed the code into the jupyter notebook `project10.ipynb`

## Usage
The code for this project can be run and followed along with by running the cells in `project10.ipynb`. 

## Pseudocode

```
Use as inputs:
    a model from our HMM module
    a sequence of emissions

init
    Given a model from our HMM module
    Select what emissionset we're training on so we're writing data to the correct table of emission probabilities
    set starting params = (transition weights, emission weights, initial weights) by method of choice
        Can be decided with pseudocounts or random number generation
        they should probably be numpy arrays throughout this whole thing for sure but they can be loaded in and ultimately returned via our classes

iteration
    
    get forward and backward probability distribution for the current sequence given current params
    get sequence likelihood (just from the forward algorithm, just backward, or average of both are all fine)
    check convergence of sequence likelihood; if threshold met then go to termination
        Don't do this if the "burn-in" period hasn't ended (checked by number of iterations)

    update params (W_hh, W_eh, W_init)
        This step is a lot and involves a lot of lookups of our weights matrices and elementwise operations that are easier when everything is numpy arrays
        Will get into the math considerations below to avoid bloating this func's pseudocode

termination
    return params (W_hh, W_eh, W_init), either raw or build a HMM with them


When converting our model's structures from dicts of lists (and similar standard python objects) to numpy arrays (do this at the start):
    Takes as input:
        The model
        The name of the emissionset being used (this is used for emission lookups, meant to accomodate a model that can handle multiple streams of paired emissions e.g. nucleotides + methylation statuses)

    make an ordered list of state names so the indices in our numpy arrays all line up
    The model should already have an ordered list of emission names ready to go so emission matrices line up

    write the P_init as a 1darray
        Fill it with each state's corrresponding initial probability value

    write P_hh (transition probs matrix) as a 2darray
        Iterate through all combinations of previous state + current state and write in the value from doing the dict lookup with those two states

    write P_eh (emission probs matrix) as a 2darray
        Here, the emission probs are already arranged in a list whose indices line up with one another
        Iterate through each state
            use the emission set name and the current state to grab the list of this state's emission probabilities


When writing the data from our numpy arrays back to the model (only do this at the end):
    Takes as input:
        The model
        the name of the emissionset
        The three numpy arrays for P_init, emission probabilities, and transition probabilities

    Iterate through the hidden state objects in the model (using enumerate to access the names and indices)
        Assign the current state's initial probability to the current index's value in the P_init array

    Just convert the transition weights array to a series of lists (since in our model objects, it's a list of lists)

    Iterate through the model's hidden state objects (using enumerate to get the index and the state itself)
        Access the current index's row of the 2darray of emission probabiliites
        Access the current hidden state object's emission weights for the current emission set
            Assign the object's emission weights to the previously-mentioned row from the 2darray


Math considerations:
number of unique hidden states = N, fixed
number of unique possible emissions = K, fixed
number of timesteps = T, fixed
sequence likelihood = Pseq, computed using forward algo or backward algo (they should return the same number)
 
theta (all the params to be optimized) is a vector/list/dict/container of length 3, holding three objects which are first defined at initiation:
- W_init, usually called pi, is a N-length vector representing our initial weights 
- W_hh, usually called A or a, is a NxN matrix representing our state transition weights 
- W_eh, usually called B or b, is a NxK matrix representing the emission weight matrix
        It is only two dimensions because the observation is fixed. 
 
probability distributions to be recomputed every iteration (i.e. for each sequence of emissions):
- The Forward distribution, usually alpha, is a TxN matrix (aka the forward algo's scoring matrix)
- forward_dist[t][i] = prob that observations up to t are true and the hidden state at t is i (basically the probability of all paths up to time t that end up at state i at time t)
- The Backward distribution, usually beta, is a TxN matrix (aka the backward algo's scoring matrix)
- backward_dist[t][i] = prob that observations after t are true given that the hidden state at t is i (basically the probability of all paths that will have come from state i at time t)
 
probability distributions to be recomputed every iteration that does not terminate after getting Pseq from the above:
- The state occupancy distribution, usually gamma, is a TxN matrix
- gamma_dist[t][i] = prob that, at time t, the hidden state is i given the observed sequence and the model's current params
                     This can be thought of as the probability of all paths that pass through state i at time t
                     Outside of log-space, this takes the form of a formula from Bayesian-stats: forward_prob(i) * backward_prob(i) / sequence_prob, for all states i at timepoint t
                     In log-space, it's just log(forward_prob(i)) + log(backward_prob(i)) - log_sequence_prob
- The transition occupancy distribution, usually ksi, is a T-1xNxN matrix:
- ksi_dist[t][i][j] = prob that, at time t, the hidden state at t is i, and the hidden state at t-1 was j, given the observed sequence and the model's current params
                      This can be thought of as the probability of all paths that go through the state transition of j at time t-1 to i at time t.
                      Out of logspace, this looks like forward_prob(i @ t) * transition_prob(i -> j) * emission_prob(j @ t+1) * backward_prob(j @ t+1) / prob(sequence)
                      In logspace, it's log_forward_prob(i @ t) + log_transition_prob(i -> j) + log_emission_prob(j @ t+1) + log_backward_prob(j @ t+1) - log_prob(sequence)

```

## Personal reflections
### Successes
Once we figured out how the math for Baum-Welch is supposed to play out and we were able to convert our HMM's data structures to numpy arrays, we were able to implement the algorithm and the associated math in a clean manner that avoids slow name lookups for states and emissionsets and time-consuming iteration for elementwise operations. 
### Strugges
It took us a bit of time to get our heads wrapped around the math used to determine the probabilities that we were trying to maximize for our expectation maximization. Once we understood the math and what everything represented, the purpose of the algorithm and the way it functions became so much clearer to us. 

We also had a very difficult time making the time to meet up to work on this project, as it coincided with the final projects for this class's lab section as well as other classess we were all taking. Thankfully, we were still able to chat over text whenever we had some time available to help each other to understand the algorithm, the math involved, and how we'd want to implement it. 
### Individual reflections
#### Linh
Most of the difficulty came from understanding the Expectation Maximization algorithm (of which Baum-Welch is a special case), as well as additive smoothing for (correctly) using pseudocounts. Expected counts, which are referred to interchangeably with "probability" and "count" in many sources, can be difficult to reason with consistently when you're out of practice. I also got a lot of practice with numpy "C-style" vectorization and broadcasting this week, which my R and Matlab experience both helped and hindered (R and Matlab use column-major order. Now I *know* I'll start messing it up the other way once I go back to R). While we set up our model constructors with only base python, it wasn't too much of a pain to convert them into numpy array format as the Baum-Welch algorithm itself only needs to get the initial parameters from the model once to initialize. 
#### Justin
I had quite the difficult time wrapping my head around Baum-Welch, however my team walked me through it and pointed me to some resources that helped me out with the mathematical notation we saw in some of the external sources I was reading. The lecture was helpful in understanding it at a bird's eye view, but it glossed over the calculations we'd need to perform every iteration to implement everything, particularly with the probabilities of a path going through a given state (gamma) or going through a given state transition (ksi). Mathematical notation isn't exactly my strongsuit, so I really appreciate my groupmates for helping me understand what I was looking at. And brushing up on the math behind the algorithm helped me understand better what was going on conceptually, as well. At first I thought that we were using Posterior-decoding to identify the most probable state at each timepoint and then using the same training scheme as though we weren't using Baum-Welch. In reality, we do still care about every state at each timepoint, and we have to do the equivalent of Excel's `countif()` function (an indicator function) to determine how often we expect to see a given emission in a given state at each timepoint, for all the states at all the timepoints. This was very satisfying to come to understand and to see this come together, especially after the headache we gave ourselves during week 8 trying to think of how we should make our models trainable. 

#### Danny
This week’s implementation of the Baum-Welch algorithm was a massive paradigm shift for me. Coming from an agricultural wet-lab background, my instinct is to look for concrete biological events, so grasping the concept of fractional "expected counts" (Gamma and Xi) across dynamic programming matrices was incredibly difficult at first. However, working with Linh and Justin to implement additive smoothing and seamlessly pass our OOP model into nice NumPy calculations finally made the math click. Watching our code train itself and increase its log-likelihood was profoundly rewarding！By the thanks for wiki wich really help a lot to understand Baum-Welch algorithm. 

## Appendix
