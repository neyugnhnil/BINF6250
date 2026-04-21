from core.EmissionSet_and_HiddenState_defs import EmissionSet, HiddenState
import numpy as np
from typing import Optional

###### WIP ###############################

# can depend on arbitrary (n) number of prior hidden states (ie. markov order)

class nthHMModel:

    ##### CLASS BASICS #####

    def __init__(self, 
        emission_sets: Optional[list[EmissionSet]] = None, 
        hidden_states: Optional[list[HiddenState]] = None, 
        markov_order: int = 1) -> None:

        # defined once
        self.markov_order = markov_order 

        # always updated
        self.emission_sets = []                                     # all legal EmissionSets
        self.hidden_states = []                                     # all actual HiddenStates

        self.W_init = None                                          # shape: (# of hs,) x (markov_order)
        self.W_trans = None                                         # shape: (# of hs,) x (markov_order + 1)
        # hs -> axis values, markov order -> number of axes

        # derived only when prompted
        self.W_eh = None
        self.P_eh = None
        self.P_init = None                                          # same shape as W_init
        self.P_trans = None                                         # same shape as W_trans

        for es in emission_sets:
            self.add_emission_set(es.copy())                        # add EmissionSet copies one at a time
        for hs in hidden_states:
            self.add_hidden_state(hs.copy(), mode="strict")         # add HiddenState copies one at a time

        # initialize empty W_init and W_trans
        self.resize_transition_tensors()

    def __repr__(self) -> str:
        return (
            f"HMModel(markov_order={self.markov_order}, "
            f"emission_sets={self.emission_sets}, "
            f"hidden_states={self.hidden_states}, "
            f"W_init_shape={None if self.W_init is None else self.W_init.shape}, "
            f"W_trans_shape={None if self.W_trans is None else self.W_trans.shape})"
        )

    def copy(self) -> "nthHMModel":
        # make a copy of current model

        new_model = nthHMModel(
            emission_sets=[es.copy() for es in self.emission_sets],
            hidden_states=[hs.copy() for hs in self.hidden_states],
            markov_order=self.markov_order
            )

        if self.W_init is not None:
            new_model.W_init = self.W_init.copy()

        if self.W_trans is not None:
            new_model.W_trans = self.W_trans.copy()

        if self.W_eh is not None:
            new_model.W_eh = {
                set_name: {
                    state_name: list(weights)
                    for state_name, weights in state_dict.items()
                }
                for set_name, state_dict in self.W_eh.items()
            }

        if self.P_init is not None:
            new_model.P_init = self.P_init.copy()

        if self.P_trans is not None:
            new_model.P_trans = self.P_trans.copy()

        if self.P_eh is not None:
            new_model.P_eh = {
                set_name: {
                    state_name: list(weights)
                    for state_name, weights in state_dict.items()
                }
                for set_name, state_dict in self.P_eh.items()
            }

        return new_model

    ##### INTERNAL CONVENIENCES #####

    def n_states(self) -> int:
        return len(self.hidden_states)

    def es_ref(self, es_ref: int | str) -> int:
        # normalizes references to an emission set in this model

        # ref is already int index
        if isinstance(es_ref, int): 
            return es_ref

        # ref is str name
        for i, emission_set in enumerate(self.emission_sets):
            if emission_set.set_name == es_ref:
                return i
        
        # invalid provided ref
        raise Exception(f"emission set {es_ref} not found")
    
    def hs_ref(self, hs_ref: int | str) -> int:
        # normalizes references to a hidden state in this model

        # ref is already int index
        if isinstance(hs_ref, int):
            return hs_ref

        # ref is str name
        for i, hidden_state in enumerate(self.hidden_states):
            if hidden_state.hidden_state_name == hs_ref:
                return i

        raise Exception(f"hidden state {hs_ref} not found")

    def get_es(self, es_ref: int | str) -> EmissionSet:
        return self.emission_sets[self.es_ref(es_ref)]

    def get_hs(self, hs_ref: int | str) -> HiddenState:
        return self.hidden_states[self.hs_ref(hs_ref)]

    def hs_name(self, hs_ref: int | str) -> str:
        return self.get_hs(hs_ref).hidden_state_name

    def normalize_history_refs(self, history_refs: tuple[int | str, ...] | list[int | str]) -> tuple[int, ...]:
        # convert a list of hidden-state refs into a tuple of indices
        
        if len(history_refs) > self.markov_order:
            raise Exception(f"history length cannot exceed markov_order={self.markov_order}")

        return tuple(self.hs_ref(ref) for ref in history_refs)

    def clear_derived(self) -> None:
        # should be called after any structural or weight change
        self.W_eh = None
        self.P_init = None
        self.P_trans = None
        self.P_eh = None

    def check_novel_es_name(self, name: str | None) -> None:
        for emission_set in self.emission_sets:
            if emission_set.set_name == name:
                raise Exception(f"duplicate emission set name: {name}")

    def check_novel_hs_name(self, name: str | None) -> None:
        for hidden_state in self.hidden_states:
            if hidden_state.hidden_state_name == name:
                raise Exception(f"duplicate hidden state name: {name}")

    def validate_hs_against_schema(self, hidden_state: HiddenState) -> None:
        # require agreement with model emission schema
        
        model_set_names = [es.set_name for es in self.emission_sets]
        state_set_names = list(hidden_state.emission_weights.keys())

        if set(state_set_names) != set(model_set_names):
            raise Exception("hidden state emission set names must exactly match model schema")

        for emission_set in self.emission_sets:
            set_name = emission_set.set_name
            if set_name not in hidden_state.emission_weights:
                raise Exception(f"missing emission set {set_name} in hidden state")
            if len(hidden_state.emission_weights[set_name]) != emission_set.length:
                raise Exception(
                    f"weight vector for emission set {set_name} must have length {emission_set.length}")

    def resize_transition_tensors(self) -> None:
        # resize W_init and W_trans when number of states changes, preserving old values
        # (this is like "adding a row and a column" when resizing the 1st order square matrix)

        K = self.n_states()

        # get expected shapes
        init_shape = (K,) * self.markov_order
        trans_shape = (K,) * (self.markov_order + 1)

        # make 0s tensors with those shapes
        new_W_init = np.zeros(init_shape, dtype=float)
        new_W_trans = np.zeros(trans_shape, dtype=float)

        if self.W_init is not None:
            # preserve old values
            old_slices = tuple(
                slice(0, min(old_dim, new_dim))
                for old_dim, new_dim in zip(self.W_init.shape, new_W_init.shape)
                ) # copy the overlap between old and new
            new_W_init[old_slices] = self.W_init[old_slices] # replace zeros 

        if self.W_trans is not None:
            # preserve old values
            trans_slices = tuple(
                slice(0, min(old_dim, new_dim))
                for old_dim, new_dim in zip(self.W_trans.shape, new_W_trans.shape)
                ) # copy the overlap between old and new
            new_W_trans[trans_slices] = self.W_trans[trans_slices] # replace zeros 

        self.W_init = new_W_init
        self.W_trans = new_W_trans

        self.clear_derived()

    ##### METHODS FOR MODIFYING THE MODEL #####

    def add_emission_set(self, new_emission_set: EmissionSet, fill_hidden_states_with: str = "zeros") -> None:
        self.check_novel_es_name(new_emission_set.set_name)
        self.emission_sets.append(new_emission_set)

        for hidden_state in self.hidden_states:
            # for hs without weight data for the emission set, assign some defaults
            if new_emission_set.set_name not in hidden_state.emission_weights:
                if fill_hidden_states_with == "zeros":
                    hidden_state.emission_weights[new_emission_set.set_name] = [0.0] * new_emission_set.length
                else:
                    hidden_state.emission_weights[new_emission_set.set_name] = list(new_emission_set.default_weights)

        self.clear_derived()

    def add_hidden_state(self, new_hidden_state: HiddenState, mode: str = "strict", missing_fill: str = "zeros") -> None:
        # add hs to model

        self.check_novel_hs_name(new_hidden_state.hidden_state_name)

        # strict = expects respected schema
        if mode == "strict":
            self.validate_hs_against_schema(new_hidden_state)

        # force = can take hidden states that do not have data for all emission sets
        if mode == "force":

            #### 1. no emission sets can be forced into the model through this route ####
            model_set_names = {es.set_name for es in self.emission_sets}
            state_set_names = set(new_hidden_state.emission_weights.keys())
            extras = state_set_names - model_set_names
            if extras:
                raise Exception(f"hidden state contains unknown emission sets: {extras}")

            #### 2. however, if the new HS is llacking emission sets, add the sets to the HS ####
            for emission_set in self.emission_sets:
                if emission_set.set_name not in new_hidden_state.emission_weights:
                    if missing_fill == "zeros":
                        new_hidden_state.emission_weights[emission_set.set_name] = [0.0] * emission_set.length
                    else:
                        new_hidden_state.emission_weights[emission_set.set_name] = list(emission_set.default_weights)

            #### 3. check if forcing was done correctly ####
            self.validate_hs_against_schema(new_hidden_state)
        
        self.hidden_states.append(new_hidden_state)
        self.resize_transition_tensors()

    def replace_hidden_state(self, hs_ref: int | str, new_hidden_state: HiddenState, mode: str = "strict") -> None:
        # replace hs by index

        idx = self.hs_ref(hs_ref)
        old_name = self.hidden_states[idx].hidden_state_name
        new_name = new_hidden_state.hidden_state_name
        
        if new_name != old_name:
            self.check_novel_hs_name(new_name)

        # strict = expects respected schema
        if mode == "strict":
            self.validate_hs_against_schema(new_hidden_state)
        
        # force = can take hidden states that do not have data for all emission sets
        if mode == "force":
            
            #### 1. no emission sets can be forced into the model through this route ####
            model_set_names = {es.set_name for es in self.emission_sets}
            state_set_names = set(new_hidden_state.emission_weights.keys())

            extras = state_set_names - model_set_names
            if extras:
                raise Exception(f"hidden state contains unknown emission sets: {extras}")

            #### 2. however, if the new HS is llacking emission sets, add the sets to the HS ####
            for emission_set in self.emission_sets:
                if emission_set.set_name not in new_hidden_state.emission_weights:
                    new_hidden_state.emission_weights[emission_set.set_name] = [0.0] * emission_set.length

            #### 3. check if forcing was done correctly ####
            self.validate_hs_against_schema(new_hidden_state)


        self.hidden_states[idx] = new_hidden_state
        self.clear_derived()

    def replace_init_weight(self, history_refs: tuple[int | str, ...] | list[int | str], new_weight: float) -> None:
        # replace init weight for a history of length=markov_order
        history_idx = self.normalize_history_refs(history_refs)

        if len(history_idx) != self.markov_order:
            raise Exception(f"init history must have length exactly {self.markov_order}")

        self.W_init[history_idx] = new_weight

        self.clear_derived()

    def replace_transition_weight(self,
        history_refs: tuple[int | str, ...] | list[int | str],
        current_state_ref: int | str,
        new_weight: float) -> None:
        # replaces one transition weight in W_trans
        
        history_idx = self.normalize_history_refs(history_refs)

        # for markov_order = m, history length must be exactly m
        if len(history_idx) != self.markov_order:
            raise Exception(f"transition history must have length exactly {self.markov_order}")

        curr_idx = self.hs_ref(current_state_ref)
        self.W_trans[history_idx + (curr_idx,)] = new_weight

        self.clear_derived()

    def replace_transition_distribution(
        self,
        history_refs: tuple[int | str, ...] | list[int | str],
        new_weights: list[float] | dict[int | str, float]
        ) -> None:
        # replace the entire next-state distribution for one history

        history_idx = self.normalize_history_refs(history_refs)

        # for markov_order = m, history length must be exactly m
        if len(history_idx) != self.markov_order:
            raise Exception(f"transition history must have length exactly {self.markov_order}")

        if isinstance(new_weights, list):
            if len(new_weights) != self.n_states():
                raise Exception("new_weights list must have length equal to number of hidden states")
            self.W_trans[history_idx] = np.array(new_weights, dtype=float)
        elif isinstance(new_weights, dict):
            row = np.zeros(self.n_states(), dtype=float)
            for curr_ref, weight in new_weights.items():
                curr_idx = self.hs_ref(curr_ref)
                row[curr_idx] = float(weight)
            self.W_trans[history_idx] = row
        else:
            raise Exception("new_weights must be a list or dict")

        self.clear_derived()

    ##### DERIVATION METHODS #####

    def build_W_eh(self) -> dict[str, dict[str, list[float]]]:
        # builds nested dictionary view of emission weights
        # ie. W_eh[emission_set_name][hidden_state_name] = [list of weights]
    
        W_eh = {}
        for es in self.emission_sets:
            set_name = es.set_name
            W_eh[set_name] = {}
            for hidden_state in self.hidden_states:
                state_name = hidden_state.hidden_state_name
                W_eh[set_name][state_name] = list(hidden_state.emission_weights[set_name])
        return W_eh

    def build_P_init(self) -> np.ndarray:
        # normalizes W_init into probabilities

        if self.W_init is None:
            raise Exception("W_init has not been initialized")

        total = self.W_init.sum()
        if total == 0:
            return np.zeros_like(self.W_init)

        return self.W_init / total

    def build_P_trans(self) -> np.ndarray:
        # normalizes W_trans over its final dimension 
        # in this model, the last dimension is the "next state" dimension ("columns" in 1st order)
        # ie. for each history, probs over the next state sum to 1

        if self.W_trans is None:
            raise Exception("W_trans has not been initialized")
        
        sums = self.W_trans.sum(axis=-1, keepdims=True)                         # sum all "outgoing" weights from each history, ie. across next states
        P_trans = np.zeros_like(self.W_trans)                                   # make output array, initially all zeroes
        np.divide(self.W_trans,sums,out=P_trans,where=(sums != 0))              # (when sums!=0) divide weights by sums

        return P_trans

    def build_P_eh(self) -> dict[str, dict[str, list[float]]]:
        P_eh = {}

        # for each emission set...
        for emission_set in self.emission_sets:
            set_name = emission_set.set_name
            P_eh[set_name] = {}

            # create weights vector for each hidden state...
            for hidden_state in self.hidden_states:
                state_name = hidden_state.hidden_state_name
                
                weights = hidden_state.emission_weights[set_name]
                
                weights_total = sum(weights)
                if weights_total == 0:
                    return [0.0] * len(weights)

                P_eh[set_name][state_name] = [w / weights_total for w in weights]

        return P_eh

    def normalize_all(self) -> None:
        self.W_eh = self.build_W_eh()
        self.P_init = self.build_P_init()
        self.P_trans = self.build_P_trans()
        self.P_eh = self.build_P_eh()

    ##### VALIDATION #####
    
    def validate_model(self) -> None:
        
        # emission sets
        seen_emission_set_names = set()
        for emission_set in self.emission_sets:

            # dupe check
            if emission_set.set_name in seen_emission_set_names:
                raise Exception(f"duplicate emission set name: {emission_set.set_name}")
            seen_emission_set_names.add(emission_set.set_name)

            # emission set internal consistency
            if emission_set.length != len(emission_set.value_names):
                raise Exception(f"emission set {emission_set.set_name} has inconsistent length/value_names")
            if emission_set.length != len(emission_set.default_weights):
                raise Exception(f"emission set {emission_set.set_name} has inconsistent length/default_weights")

        # hidden states
        seen_hidden_state_names = set()

        for hidden_state in self.hidden_states:
            # dupe check
            if hidden_state.hidden_state_name in seen_hidden_state_names:
                raise Exception(f"duplicate hidden state name: {hidden_state.hidden_state_name}")
            seen_hidden_state_names.add(hidden_state.hidden_state_name)

            # against schema
            self.validate_hs_against_schema(hidden_state)

        # check expected shape of init and transition tensors
        
        K = self.n_states()
        expected_init_shape = (K,) * self.markov_order
        expected_trans_shape = (K,) * (self.markov_order + 1)

        if self.W_init is None or self.W_init.shape != expected_init_shape:
            raise Exception(f"W_init must have shape {expected_init_shape}")

        if self.W_trans is None or self.W_trans.shape != expected_trans_shape:
            raise Exception(f"W_trans must have shape {expected_trans_shape}")