# ProfileHMM class method that returns transition probs and emission probs arrays
# assumes global alignment input

import numpy as np
from typing import List


def get_profile_hmm_probs(
    msa_input: List[List],
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

### test ###


# M_STATE, I_STATE, D_STATE, SKIP_STATE = 0, 1, 2, 3

# msa_listlist = [
#     list("A-C"),
#     list("TGC"),
#     list("--C"),
#     list("--C")
#     ]

# trans_probs, emit_probs = get_profile_hmm_probs(
#     msa_input=msa_listlist,
#     alphabet=list("ACTG"),
#     match_threshold=0.5,
#     pseudocount=1,
#     gap="-"
#     )

