# Introduction
This project is our team's implementation of a profile HMM that acts as an extension of the `BASEHMM` class provided in the assignment. This requires our profile HMM to be compatible with the base code of the `BASEHMM` class, which is meant as a generalized class that handles any kind of HMM, as opposed to being specifically tailored for profile HMMs. 

With our profileHMM class, we train two separate profile HMMs off of the aligned sequences in each of the motif fasta files (`phmm_train_motif1.fasta` and `phmm_train_motif2.fasta`) and use the two models to attempt to calculate the probability of the sequences from `phmm_test_sequences.fasta` containing either of the the two motifs. 

# Pseudocode

```
Define our Profile HMM class as a child of BaseHMM
If the user wants to use Viterbi and other algos on this, pass this model into the HMM class

Parse FASTA to get the MSA, then pass it to the pHMM class

__init__
inputs:
	MSA
	Alphabet
	gap symbol = "-"

	Call helper funcs to define emission and trans probs as np arrays, based on MSA (currently encapsulated by main())
		These implicitly tell you how many profile positions (aka match positions) there are through their shape
	Use the shape of one of these arrays to make a list of our hidden states
	Call helper funcs to convert np arrays to dicts
	Make a dict for init probs (make it a default dict so we have {m0: 1, everyone else 0}, because m0 is actually the beta but we want it on the matrix, but it doesn't have an emission)
		This can be a comprehension

	Call super().__init__ and pass in the dicts for emission, trans, and init probs
		Pass in the alphabet and add the gap symbol to the alphabet so baum welch and others won't break
		Pass in the states too

Helper funcs:

Generate all the states:
Input: number of match positions in alignment (this will be inferred from the shape of the 3darrays for emission and/or transition probabilities)
	Text manipulation, it's just a list of these:
		m0
		m{i}
		i{i-1}
		d{i}
	for all i in the range from 1 to length of alignment + 1


Convert trans probs from arrays to dictlikes:
	Input: 3d array of transitions where:
			Outer array is match positions
			Middle array is state we're coming from, indexed matching ordered list of match, insert, delete
			Inner array is state we're going to, indexed matching above
			Values in inner array are trans_prob(middle_array_state --> inner_array_state)
	
	For i, array_i in enumerate(outer array) (represents match position aka profile position)
		
		for j, row in enumerate(array_i) (represents source state)
			make the key for the source state 
				use row number j to index the name of what we're doing out of M, I, and D
				Concat that to i the index

			initialize dict[source_state] as defaultdict with default value of 0

			dict[source_state]["M{i+1}"] = row[0]
				(the key "source_state" encapsulates the position, because it's always formatted as <letter> + <i>)
			dict[source_state]["I{i}"] = row[1]
			dict[source_state]["D{i+1}"] = row[2]

	Now we return our dict, it's set up such that: probability = dict[source][destination]


Convert emit probs from arrays to dictlike:
	Input: 3d array of type of position, match position, and alphabet character
			Index for outer array represents match position, i
			Index for middle array represents whether you're looking at match states M{i} or insertion states I{i}
			Index for inner array represents which emission

	initialize our dict {}

	Get a dict for the background emission probs for our insertion states: 3darray[i][1], pick any i
		{alphabet[a]: background_probs[a] for a in range(len(alphabet))}
		This comprehension is just {emission: background prob of emission taken from any of the insertion states' values on the 3darray}

	Do the same thing as above but set all the probs to 0, this is our emission probs for every deletion state
		{emission: 0 for emission in alphabet}

	Iterate over the match positions i
		3darray[i][0] will be the match state one, make this a dict
		3darray[i][1] will be the insertion state one, it's the same as all the other I{i} emission probs, so we don't need to unpack
		3darray[i][2] will be the deletion one, we don't need to unpack this either

		Just do this:
		dict["M{i}"] = match_state_dict  # this is the only one that actually changes every iteration
		dict["D{i}"] = deletion_state_dict
		dict["I{i}"] = insertion_state_dict
		
	We've now made a dict formatted like so:
	{M{i}: {A: , B: , C: , D: }, I{i}: {A: B: C: D: }, D{i}: {any: 0}}

	
			

Things we've already written (so the pseudocode for this can just come from Linh's code comments):

Encapsulated within main:

	Get emission counts for all states (using np arrays)

	Get trans counts for all states (using np arrays)

Called once a piece within main:

	Convert emission counts to probabilities, for all states (using np arrays)

	Convert transition counts to probabilities, for all states (using np arrays)



== In our notebook demo/main ==

Do this for both training files:
	Read in the aligned seq's in the files
	Train a pHMM on that alignment
	convert the pHMM from a pHMM object to an HMM object (from the code we'd been given) so that it gains the functions for running forward, Viterbi, etc etc
	use BASEHMM.tojson to print the model's probabilities to output files

Read in the seq's in the test file
for each sequence in the test file:
	Go through each HMM
		Use forward algo to get probability of sequence under current pHMM
		Use Viterbi to figure out where matches, deletions, insertions are likely to be in this sequence (this is what the optimal state path represents)

	Print out the outputs for this gene
```

# Successes
Using a pre-defined HMM module with pretty intuitive datastructures and algorithms like Viterbi and Forward-Backward already programmed definitely made our lives easier. We had a much easier time understanding how everything needed to work in our pHMM module once we broke down how the architecture worked. Once we figured out the architecture for a pHMM, it made it pretty straightforward for us to understand how that related to a normal (non-profile) HMM's datastructures and how we could adapt the generlized HMM module we were given to accomodate the high volume of states in a profile HMM. 

# Struggles
We found it somewhat difficult to figure out how we should handle the calculations for the emissions and transitions, as the architecture for the profile HMM was a little unintuitive at first. We had to be explicit about how the number in each state's name referred to the number of the accompanying match state, which was also the position in the consensus sequence. We also had to explicitly write out where each state at position i could go to. Once we had an idea of what state transitions were legal, what each position (i) represented, and what each state transition represented, it made it much easier for us to understand how we needed to build the profile HMM. 

While we were able to figure out the aspects of designing the profile HMM that were giving us trouble, we ran into integration issues when we tried passing our profile-HMM objects to the given HMM class so we could use Viterbi and the other algorithms. The issues seem to lay in the assumptions that BaseHMM makes about the data structures it expects our model to have, and it therefore throws errors associated with BaseHMM's validation functions. At a glance, though, it doesn't seem like the assumptions the class makes are violated. For instance, it has been throwing an error stating that all transition probabilities for leaving a state have to add to 1, yet we already made sure of that when we calculated the transition probabilities, as we made sure that the count of all transitions from a given source to a given destination is divided by the sum of the counts for that source going to each of its possible destinations. In other words, the validation functions are throwing errors for things that should already be fine. This has left us successful in creating the profile-HMM class as a child of the BaseHMM class but unsuccessful in actually using it with the functions in the HMM class. Conceptually, we've got the profile HMM down, but we're getting trolled in our implementation by what we can only assume to be inconspicuous indexing or math oversights. 

# Personal Reflections
## Group Leader: Linh
We should have looked at the BaseHMM/HMM assumptions earlier. We focused a lot on defining the Profile HMM and specifically learning about Plan 7. This was a good way to notice how a Profile HMM is just a special case of a HMM, with a lot of 0 probabilities. We tried to use defaultdict, but ran into some hard assumptions of BaseHMM because of that, so we ended up assigning 0 probabilities with dict comprehension in the end. We ended up being able to create the ProfileHMM child class while only silencing 2 validation blocks in the source code (emission probabilities being required to add up to 1; Plan 7 style models have definitionally silent states), but didn't have enough time to figure out how to make something HMM() accepts in place of BaseHMM (we probably just need to get a clearer picture of how inheritance works in this case).

## Danny
We hit a major roadblock trying to pass our ProfileHMM parameters into the BaseHMM class. We noticed that BaseHMM's @hidden_states.setter forcefully concatenates state lists into a single string, which completely breaks our model since our state names are multi-character strings (like 'M1', 'I2').
Despite all these frustrating bugs, figuring this out together has been an incredibly fun exploration process! I've learned so much that I didn't know before. For instance, this was my very first time actually understanding and applying class inheritance in Python—specifically learning how we can use it to hijack and bypass buggy initialization code while keeping all the good methods.

## Justin
I think I found the setup for the profile HMM to be the most difficult, with determining the numbering on the match and insertion states and what we gained from labelling all of the states like in the html demo Marcus provided us. It was particularly unintuitive that we didn't just have one match, one insertion, and one deletion state for each position in the alignment (from 0 to N, where N is the alignment length). It began to click for me, though, when I realized that the numbering associates with the positions from the consensus sequence, which we figure out when we're classifying positions in the alignment as either matches or insertions. Developing the language to describe these positions vs the positions in the original alignment was also a little awkward, though we settled on caling these positions the profile positions, with the associated length being called the profile's length to distinguish it from the alignment length. From there, figuring out how the calculations for the emission and transition probabilities work was a bit easier, and we just had to spend some time figuring out how to adapt our original approach (where we used numpy arrays to make the operations easier) to the dict structure of the `BASEHMM` class we've been given. This project was quite daunting at the start, but my team helped defuse a lot of that worry and it ended up being pretty fun. 

# Generative AI Appendix
We did not use generative AI on this project. 
