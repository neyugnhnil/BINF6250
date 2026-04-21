# this works for creating a BaseHMM-inheriting class if you comment out the "emission probs should add to 1" checks in BaseHMM, as well as the part where it turns the list of hidden states into a string (???)

# import statements
import numpy as np
from collections import defaultdict
from assignment_materials.HMM import BaseHMM, HMM


class profile_HMM(BaseHMM):

    def __init__(self, alignment: list[list[str]],
                 alphabet: list[str],
                 gap: str = "-",
                 seed = None, 
                 precision = 2, 
                 tolerance = 1e-10):
        """
        Initializes the profile HMM obejct

        Parameters
        ----------
        alignment : list[list[str]]
            multiple sequence alignment formatted as list of list of strings
        alphabet : list[str]
            list of legal emissions
        gap : str, optional
            character used in the multiple sequence alignment to represent a gap in one seq
            by default "-"
        """

        # call helper funcs to define emission and trans probs as np arrays
        trans_probs, emit_probs = self.get_profile_hmm_probs(
            msa_input=alignment, 
            alphabet=alphabet, 
            gap=gap
            )

        # this list represents the order that match, insertion, and deletion states are written into the 3d arrays
        self.namelist = ["M", "I", "D"]

        # add the gap character to the alphabet so Viterbi and others don't break later
        alphabet_with_gap = alphabet + [gap]

        # use the shape of our emission probs array to know how many match positions we have (aka how many 
        # profile positions to have). 
        # The idices of the outer array on both trans_probs and emit_probs represent the match positions
        model_length = emit_probs.shape[0]
        self.hidden_states = self._make_states_list(model_length)

        # conver the np arrays to dicts
        trans_dict = self._trans_probs_to_dict(trans_probs)
        emit_dict = self._emit_probs_to_dict(emit_probs, alphabet, gap)

        # make a dict for init probs (we're putting M0 on the transition matrix so this is easier)
        init_probs = {key:0.0 for key in self.hidden_states}
        init_probs["M0"] = 1.0

        # now call the initialization code from the parent class so our other attributes are up and running
        super().__init__(
            alphabet=alphabet_with_gap, 
            hidden_states=self.hidden_states, 
            init_probs=init_probs, 
            trans_probs=trans_dict, 
            emit_probs=emit_dict, 
            seed=seed, 
            precision=precision, 
            tolerance=tolerance
            )

    
    def get_profile_hmm_probs(
        self,
        msa_input: list[list],
        alphabet: list = list('ATCG'),
        match_threshold: float = 0.5,
        pseudocount: float = 1,
        gap: str = "-"
        ):

        #### PREP ####
        # covert list-of-lists into 2D unicode array
        msa = np.array(msa_input, dtype="U1")                                           # (N, n_cols)
        if msa.ndim != 2:
            raise ValueError("msa_input rows must have same length.")
        N, n_cols = msa.shape
        # N is the number of sequences

        # resolve alphabet
        invalid_chars = set(np.unique(msa)) - {gap} - set(alphabet)
        if invalid_chars:
            warnings.warn(
                f"chars {invalid_chars} appear in the MSA but are not in the alphabet; treating as gap",
                UserWarning
                )
        A = len(alphabet)

        #### IDENTIFY MATCH COLUMNS ####
        non_gap_frac = np.mean(msa != gap, axis=0)           # get non-gap fraction for each alignment position (axis 0)
        match_col_mask = non_gap_frac >= match_threshold     # boolean mask for match columns
        
        L = int(match_col_mask.sum())
        if L == 0:
            raise ValueError(f"no match columns found with match_threshold={match_threshold}!")

        # col_profile_pos[c] = cumulative match-col count up to and including col c
        # for mapping alignment column <-> profile position
        col_profile_pos = np.cumsum(match_col_mask).astype(int)

        #### ENCODE MSA ####
        # for every "cell" of msa: emission index in alphabet, or -1 for gap
        char_idx_matrix = np.full((N, n_cols), -1, dtype=int)                   # (N, n_cols)
        for a, ch in enumerate(alphabet):
            char_idx_matrix[msa == ch] = a

        ### ASSIGN EACH CELL A STATE ###
        is_gap_mat = char_idx_matrix < 0                    # (N, n_cols)
        state_matrix = np.where(
            match_col_mask[np.newaxis, :],                  # (n_cols,) promoted to (1, n_cols)
            np.where(is_gap_mat, D_STATE, M_STATE),         # within a match column: gap -> deletion, character -> match
            np.where(is_gap_mat, SKIP_STATE, I_STATE)       # within a non-match column: gap -> skip, character -> insert
            ).astype(int)
        # np.where broadcasts all of its arguments to a common shape before applying the condition elementwise
        # the outer np.where ask "where are the match columns" accross MSA columns
        # the inner 2 np.wheres ask "is there a gap or a character" accross sequences at that column
        
        ### GET MATCH EMISSION COUNTS ###    
        # prep (L + 1, A) shape
        match_emit_counts = np.zeros((L + 1, A))

        for col in np.nonzero(match_col_mask)[0]:
            pos = col_profile_pos[col]
            for char_idx in char_idx_matrix[:, col]:
                if char_idx >= 0:
                    match_emit_counts[pos, char_idx] += 1

        ### GET INSERT EMISSION COUNTS ###
        # prep (A,) shape
        insert_emit_counts = np.zeros(A)

        for col in np.nonzero(~match_col_mask)[0]:
            for char_idx in char_idx_matrix[:, col]:
                if char_idx >= 0:
                    insert_emit_counts[char_idx] += 1

        ### GET TRANS COUNTS ###
        # prep (L + 1, 3, 3) shape
        trans_counts = np.zeros((L + 1, 3, 3))

        for n in range(N):      # for each sequence in MSA...
            # which MSA cols were non-skip?
            valid_cols = np.nonzero(state_matrix[n] != SKIP_STATE)[0]
            if valid_cols.size == 0:
                # this sequence is somehow all skip states
                continue
            
            path = [(0, M_STATE)]  # start state M0
            
            for col in valid_cols:
                pos = col_profile_pos[col]
                state = state_matrix[n, col]
                path.append((pos, state))

            for (source_pos, source_state), (_, dest_state) in zip(path[:-1], path[1:]):
                trans_counts[source_pos, source_state, dest_state] += 1

        ### COMPUTE TRANS PROBS ###
        # add pseudocount
        raw = trans_counts + pseudocount

        # Plan7 transitions:
        # M_i -> M_{i+1}, I_i, D_{i+1} for i = 0..L-1
        # I_i -> I_i or M_{i+1} for i = 0..L-1, including I_0
        # D_i -> M_{i+1} or D_{i+1} for i = 1..L-1
        # D_0 is not used.
        allowed = np.zeros_like(raw, dtype=bool)
        allowed[:L, M_STATE, :] = True  
        allowed[:L, I_STATE, M_STATE] = True 
        allowed[:L, I_STATE, I_STATE] = True
        allowed[1:L, D_STATE, M_STATE] = True
        allowed[1:L, D_STATE, D_STATE] = True

        # remove pseudocounts from raw where not allowed
        raw[~allowed] = 0

        # normalize along the last axis (destination state) 
        destination_sums = raw.sum(axis=-1, keepdims=True)          # (L+1, 3, 1)
        destination_sums[destination_sums == 0] = 1.0               # this is to make sure 0s stay 0                          
        trans_probs = raw / destination_sums

        ### COMPUTE EMISSION PROBS ####
        # M states have one distribution per profile position.
        raw_match = match_emit_counts + pseudocount                         # (L+1, A)
        match_probs = raw_match / raw_match.sum(axis=-1, keepdims=True)     # (L+1, A) / (L+1, 1) = (L+1, A)

        # I states get single shared background distribution.               
        raw_insert = insert_emit_counts + pseudocount                       # (A,)
        insert_probs = raw_insert / raw_insert.sum()                        # (A,)

        # note: the i=0 row (start) should have had no counts added
        # so at that position we created a meaningless uniform distribution of emissions (thanks to pseudocount)
        # which is fine

        # build emit_probs array
        emit_probs = np.zeros((L + 1, 2, A))           # prep shape (L+1, A)
        emit_probs[:, M_STATE, :] = match_probs        # exact copy of match_probs 
        emit_probs[:, I_STATE, :] = insert_probs       # (A,) broadcasts to all (L+1, A)

        return trans_probs, emit_probs

    def _make_states_list(self, model_length: int) -> list[str]:
        """
        Function to generate a list of the states for a profile HMM of
        the given length

        Parameters
        ----------
        model_length : int
            number of match positions in the multiple sequence alignment used
            to make this profile HMM

        Returns
        -------
        list[str]
            List of the states, includes M0 
            and M{i}, I{i-1}, and D{i} for all i from 1 to model_length
        """
        # make a list of M{i} for all i from 0 to model_length
        matches = [f"M{i}" for i in range(model_length + 1)]

        # ditto for D{i} but starting at 1
        deletions = [f"D{i}" for i in range(model_length + 1)]

        # ditto for I{i-1}, again starting at 1
        insertions = [f"I{i}" for i in range(model_length + 1)]

        return matches + deletions + insertions
    
    def _trans_probs_to_dict(self, trans_probs: np.ndarray) -> dict[str, dict[str, float]]:
        """
        Function to convert transition probabilities matrix 
        from Numpy 3darray to dictionaries

        Parameters
        ----------
        trans_probs : np.ndarray
            3d numpy array where:
                Outer array indices represents match positions
                Middle array is state we're coming from, indexed matching ordered list of match, insert, delete
                Inner array is state we're going to, indexed matching above
            and Values in inner array are trans_prob(middle_array_state --> inner_array_state)

        Returns
        -------
        dict[str, dict[str, float]]
            dict mapping source state to a dict that, itself, maps destination state to
            the probability P(source -> destination)
        """
        # initialize output
        trans_dict = {
            source:{destination:0.0 for destination in self.hidden_states} 
            for source in self.hidden_states}
        
        #print("beep",trans_dict["M0"])

        # iterate over the profile positions
        for i, array_i in enumerate(trans_probs):

            for j, row in enumerate(array_i):
                # make the key for the source state
                source_state = self.namelist[j] + str(i)

                # initialize as a defaultdict here so we correctly report 0 probability for illegal transitions (e.g. D4 to M0)
                # trans_dict[source_state] = defaultdict(float)

                # now transition probabilities are listed in the current row, in order of M{i+1} at index 0
                trans_dict[source_state][f"M{i+1}"] = row[0]

                # I{i} is always at index 1
                trans_dict[source_state][f"I{i}"] = row[1]

                # and D{i+1} is always at index 2
                trans_dict[source_state][f"D{i+1}"] = row[2]
        
        #print("boop",trans_dict["M0"])
        return trans_dict

    def _emit_probs_to_dict(self, emit_probs: np.ndarray, alphabet: list[str], gap: str = "-") -> dict[str, dict[str, float]]:
        """
        Function to convert the emission probability matrix from a numpy 3darray
        to a dict of dicts

        Parameters
        ----------
        emit_probs : np.ndarray
            numpy 3darray for emission probabilities, where
                Index for outer array represents match position, i
			    Index for middle array represents whether you're looking at match states M{i} or insertion states I{i}
			    Index for inner array represents which emission
        alphabet: list[str]
            List of legal emissions
        gap: str, optional
            The character used to represent a gap in an aligned sequence, set as
            the guaranteed emission of the Deletion states, D{i} and has a probability of 0
            in insertion and match states
            "-" by default

        Returns
        -------
        dict[str, dict[str, float]]
            dict mapping state to a dict that, itself, maps emission name to P(emission | State)
        """
        # initialize our output
        emit_dict = {
            state:{emission:0.0 for emission in alphabet} 
            for state in self.hidden_states}

        # get a dict for the background emission probabilities to reuse for our insertion states
        # at any position, i, we can use index 1 for our middle array and it always gives us the same inner array
        background_array = emit_probs[0,1]
        background_dict = {alphabet[a]: background_array[a] for a in range(len(alphabet)) if alphabet[a] != gap}
        background_dict[gap] = 0.0

        # set up the same thing for the deletion states, except those aren't on emit_prob and have 
        # a probability of 0 for every emission
        deletion_dict = {emission: 0.0 for emission in alphabet}
        deletion_dict[gap] = 1.0

        # iterate over match positions
        for i in range(len(emit_probs)):
            
            # emit_probs[i][0] is always the match state array for this position
            # unpack the match state array into a dict
            match_dict = {alphabet[a]: emit_probs[i, 0, a] for a in range(len(alphabet))}
            match_dict[gap] = 0.0

            # now emit_probs[i][1] is always the insertion state array, which is always the same
            # so the insertion state array doesn't need to be unpacked again

            # now get all the states' entries written up
            emit_dict[f"M{i}"] = match_dict
            emit_dict[f"D{i}"] = deletion_dict
            emit_dict[f"I{i}"] = background_dict

        return emit_dict


M_STATE, I_STATE, D_STATE, SKIP_STATE = 0, 1, 2, 3

msa_listlist = [
    list("A-C"),
    list("TGC"),
    list("--C"),
    list("--C")
    ]

phmm = profile_HMM(alignment=msa_listlist,alphabet=list("ATCG"),gap="-")

print(phmm.alphabet)
print(phmm.hidden_states)
print(phmm.trans_probs['M2'])
print(phmm.emit_probs['M2'])
