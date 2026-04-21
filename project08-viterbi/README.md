# BINF6250 Project 8: Viterbi Algorithm

## Description of the project
This repository contains work done as part of the BINF6250 Algorithmic Foundations in Bioinformatics course (Northeastern University, Spring 2026).

This one's about the Viterbi algorithm for inferrence of the most likely sequence of hidden states, but mostly we were messing around with our model constructor.

## Contents

- this `README.md`
- `project08.ipynb`

## Pseudocode (for the Viterbi algorithm)

```
n = len(emissions)

states = [all possible hidden states in index order] 

scores = (len(states) x n) matrix of -inf

traceback = (len(states) x n) matrix of nones

for s in states:
	scores[0][s] = init probability of s * P(emission[0]|s)

from i in (emission[1] to emission[n]):

	for curr_s in states:
		best_score = scores[i][curr_s]
		best_prev = traceback[i][curr_s]
		emit_p = P(i|curr_s)
		
		for prev_s in states:
			prior = P(prev_s) = scores[i-1][prev_s]
			trans_p = P(prev_s -> curr_s)
			score = prior * trans_p * emit_p
			
			if score > best_score:
				best_score = score
				best_prev = prev_s
		
		scores[i][curr_s] = best_score
		
		traceback[i][curr_s] = best_prev

reverse_path = [states[argmax(scores[n])]]

for i in emission[n] to emmission[0]:
	
	state_at_i = reverse_path[-1]
	where_from = traceback[i][state_at_i]
	reversed_path.append(where_from)
	
return reversed(reverse_path)
```

In the implementation used in the notebook, we don't keep track of the entire lookup table of scores.

## Personal reflections
### Successes
We were able to come up with a very robust schema for how our Hidden Markov Model objects should be structured, which should minimize any modificaitons we need to make in the coming weeks to accomodate projects 9 and 10. Additionally, we set up our HMM module in such a way that the lookups needed to run Viterbi were pretty easy to make, and our Viterbi script was about as complicated in implementation as it was in our simplified pseudocode (beyond needing to add checks to ensure that the data passed into the algorithm was compatible with the model with which the algorithm was being told to analyze the data). 

### Strugges
Planning out our HMM module was very time consuming, and it involved a lot of trying to essentially predict what weird edge cases we might be handed in future projects. Our solution to that was to give our data structures myriad checks to ensure that the data they are being handed is of the correct type and to add guardrails that keep the user from doing things that might break the model (such as adding weights to the transition matrix after the transition probabilities had been calculated, which we handled by clearing all probability matrices every time data is added to the weight matrices). Justin had brought up the idea of implementing a method to train a model at whatever order we want, which led the whole team down a rabbit hole regarding how we generalize an HMM to Nth order, what each of those states represents, different approaches for going to Nth order, whether information is lost, if the order of the model affects compatibility with the algorithms we want to apply, and concerns about if the Markov model even needs to be able to do this for our purposes in the class. The answer wound up being that we don't need to generalize our models to Nth order, which saves us a lot of headache in terms of actual implementation, though it feels somewhat unsatisfying to have to leave those ends loose. From a learning standpoint, it was still very productive for us to go down that rabbit hole, even if, practically speaking, it took away a lot of time from implementing the Viterbi algorithm. 

### Individual reflections
#### Linh: 
This project has been interesting as we've all been able to explore the specificities of HMM in particular and of Markovness in general. Implementing the Viterbi algorithm seemed fairly straightforward, so we filled the space with an initiative to design an extensible setup for future use, which we will likely benefit from. We also attempted to generalize the model construction itself for arbitrary order of Markov property, as well as extend HMM in directions such as "additionally, condition state probabilities on previous observations". This latter example is conceptually reasonable but would require major mods to classic HMM if we wanted to stick with HMM structure/methods; nothing unprecedented, but the kind of modeling choices you'd make for a specific purpose. We may just be forgetting that models are limited to adressing certain types of questions and designs, and as these projects are meant to teach us the basic algorithmic toolkit for inferrence with HMMs, we should be wary of trying to depart from its most basic assumptions and definitions for no reason. As usual I appreciate the learning and group discussion that the rabbit hole has wrought. 

#### Justin:
I think that a lot of this week's difficulty came from needing to write the extensible architecture for our HMM module, with the viterbi algorithm itself taking up a minority share of our time spent on this week's project. We took a lot of time trying to break our brains trying to think of how to make our HMM module as flexible as possible, which included attempting to generalize it to Nth order. As it would turn out, we didn't need to do that, so the hours we spent trying to plan that could have been spent elsewhere. I think it was still interesting to think of how we can generalize to Nth order, especially because Linh brought up the idea of always training our model at 1st order and then computing Nth order probabilities later on by using a tensor, which was a new concept for me (just an ndarray of however many dimensions it needs to compute the probabilities of any kmer occuring if the emissions are nucleotides, for example). After days of planning out and scripting our module for Hidden Markov Models as objects, the Viterbi algorithm took us a day and like half an hour of planning. Additionally, the fact that we had to frontload all the work for making our HMM module to this past week means that we're freed up significantly to focus more on the algorithms themslves. I look forward to the next couple of weeks, as I was passingly familiar with posterior decoding from Genomics, and I'm really curious to see what Baum-Welch has in store for us. 

Hongyuan Deng:Oriented Programming. We navigated complex architectural choices, balancing the readability of dictionaries against the computational speed of NumPy tensors for Viterbi decoding. Through collaboration, we designed a scalable architecture supporting emission sets for future extensibility. This experience deepened my understanding of dynamic programming, supervised training in bioinformatics.

## Appendix
No AI use
