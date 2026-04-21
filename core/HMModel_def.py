from typing import Optional

from .EmissionSet_and_HiddenState_defs import EmissionSet, HiddenState

class HMModel:

    def __init__(
        self,
        emission_sets: Optional[list[EmissionSet]] = None,
        hidden_states: Optional[list[HiddenState]] = None
        ) -> None:

        emission_sets = [] if emission_sets is None else emission_sets
        hidden_states = [] if hidden_states is None else hidden_states

        # always updated
        self.emission_sets = []                                 # all legal EmissionSets
        self.hidden_states = []                                 # all HiddenStates
        self.W_hh = []                                          # transition probability matrix (HiddenState -> HiddenState)

        # cached lookup objects
        self.es_lookup = {}
        self.hs_lookup = {}

        # not computed until prompted
        self.W_eh = None
        self.P_init = None
        self.P_hh = None
        self.P_eh = None

        for es in emission_sets:
            self.add_emission_set(es.copy())                    # add EmissionSet copies one at a time
        for hs in hidden_states:
            self.add_hidden_state(hs.copy(), mode="strict")     # add HiddenState copies one at a time

    def __repr__(self) -> str:
        # represents itself by its init definition
        return (
            f"HMModel(emission_sets={self.emission_sets}, "
            f"hidden_states={self.hidden_states}, "
            f"W_hh={self.W_hh})"
            )

    def copy(self) -> "HMModel":
        new_model = HMModel(
            emission_sets=[es.copy() for es in self.emission_sets],
            hidden_states=[hs.copy() for hs in self.hidden_states]
            )

        new_model.W_hh = [list(row) for row in self.W_hh]

        if self.W_eh is not None:
            new_model.W_eh = {
                set_name: {
                    state_name: list(weights)
                    for state_name, weights in state_dict.items()
                }
                for set_name, state_dict in self.W_eh.items()
            }

        if self.P_init is not None:
            new_model.P_init = dict(self.P_init)

        if self.P_hh is not None:
            new_model.P_hh = {
                prev_state: dict(curr_dict)
                for prev_state, curr_dict in self.P_hh.items()
            }

        if self.P_eh is not None:
            new_model.P_eh = {
                set_name: {
                    state_name: list(weights)
                    for state_name, weights in state_dict.items()
                }
                for set_name, state_dict in self.P_eh.items()
            }

        return new_model

    def rebuild_lookup_objects(self) -> None:
        # rebuild cached name to index lookup dictionaries to avoid repeated scanning
        self.es_lookup = {
            emission_set.set_name: i
            for i, emission_set in enumerate(self.emission_sets)
        }
        self.hs_lookup = {
            hidden_state.hidden_state_name: i
            for i, hidden_state in enumerate(self.hidden_states)
        }

    def es_ref(self, es_ref: int | str) -> int:
        # normalizes references to an emission set in this model

        # ref is already int index
        if isinstance(es_ref, int): 
            return es_ref

        # ref is str name
        if es_ref in self.es_lookup:
            return self.es_lookup[es_ref]
        
        # invalid provided ref
        raise Exception(f"emission set {es_ref} not found")

    def hs_ref(self, hs_ref: int | str) -> int:
        # normalizes references to a hidden state in this model

        # ref is already int index
        if isinstance(hs_ref, int):
            return hs_ref

        # ref is str name
        if hs_ref in self.hs_lookup:
            return self.hs_lookup[hs_ref]

        raise KeyError(f"hidden state {hs_ref} not found")

    def get_es(self, es_ref: int | str) -> EmissionSet:
        # return emission set object by ref
        return self.emission_sets[self.es_ref(es_ref)]

    def get_hs(self, hs_ref: int | str) -> HiddenState:
        # return hidden state object by ref
        return self.hidden_states[self.hs_ref(hs_ref)]

    def clear_derived(self) -> None:
        # clear any derived weight/probability objects
        self.W_eh = None
        self.P_init = None
        self.P_hh = None
        self.P_eh = None

    def check_novel_es_name(self, name: str | None) -> None:
        # is this ES name already taken?
        if name in self.es_lookup:
            raise Exception(f"duplicate emission set name: {name}")

    def check_novel_hs_name(self, name: str | None) -> None:
        # is this HS name already taken?
        if name in self.hs_lookup:
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
                raise Exception(f"weight vector for emission set {set_name} must have length {emission_set.length}")

    def force_fill_hs_emissions(self, hidden_state: HiddenState, missing_fill: str = "zeros") -> None:
        # fill missing emission sets in a hidden state while rejecting unknown ones

        model_set_names = {es.set_name for es in self.emission_sets}
        state_set_names = set(hidden_state.emission_weights.keys())

        extras = state_set_names - model_set_names
        if extras:
            raise Exception(f"hidden state contains unknown emission sets: {extras}")

        for emission_set in self.emission_sets:
            if emission_set.set_name not in hidden_state.emission_weights:
                if missing_fill == "zeros":
                    hidden_state.emission_weights[emission_set.set_name] = [0.0] * emission_set.length
                else:
                    hidden_state.emission_weights[emission_set.set_name] = [float(w) for w in emission_set.default_weights]

    def enlarge_transition_matrix(self) -> int:
        # enlarge W_hh by one row and one column, returning old size

        old_n = len(self.hidden_states) - 1

        # add new column (= lengthen each row)
        for row in self.W_hh:
            row.append(0.0)

        # add new row
        self.W_hh.append([0.0] * (old_n + 1))

        return old_n

    def add_emission_set(self, new_emission_set: EmissionSet, fill_hidden_states_with: str = "zeros") -> None:

        # add an emission set to the model schema
        self.check_novel_es_name(new_emission_set.set_name)
        self.emission_sets.append(new_emission_set)
        self.rebuild_lookup_objects()

        for hidden_state in self.hidden_states:
            # give hidden states with no wegihts for this emission set some default values
            if new_emission_set.set_name not in hidden_state.emission_weights:
                if fill_hidden_states_with == "zeros":
                    hidden_state.emission_weights[new_emission_set.set_name] = [0.0] * new_emission_set.length
                else:
                    hidden_state.emission_weights[new_emission_set.set_name] = [float(w) for w in new_emission_set.default_weights]

        # this changes everything so clear derived
        self.clear_derived()

    def add_hidden_state(
        self,
        new_hidden_state: HiddenState,
        mode: str = "strict",
        missing_fill: str = "zeros",
        incoming_transition_weights: Optional[list[float] | dict[str, float]] = None,
        outgoing_transition_weights: Optional[list[float] | dict[str, float]] = None,
        self_transition_weight: float = 0.0,
        update_init_weight: Optional[float] = None
        ) -> None:

        # add a hidden state and enlarge transition matrix
        if mode not in {"strict", "force"}:
            raise Exception("mode must be 'strict' or 'force'")
        if missing_fill not in {"zeros", "default"}:
            raise Exception("missing_fill must be 'zeros' or 'default'")

        self.check_novel_hs_name(new_hidden_state.hidden_state_name)

        # strict = expects respected schema
        if mode == "strict": 
            self.validate_hs_against_schema(new_hidden_state)

        # force = can take hidden states that do not have data for all emission sets
        if mode == "force":

            #### 1. no emission sets can be forced into the model through this route ####
            #### 2. however, if the new HS is LACKING emission sets, add the sets to the HS ####
            self.force_fill_hs_emissions(new_hidden_state, missing_fill=missing_fill)

            #### 3. check if forcing was done correctly
            self.validate_hs_against_schema(new_hidden_state)

        # option to specify init weight for this hs in this model
        if update_init_weight is not None:
            new_hidden_state.init_weight = float(update_init_weight)

        #### MATRIX ENLARGEMENT PROCEDURE ####
        self.hidden_states.append(new_hidden_state)
        self.rebuild_lookup_objects()
        old_n = self.enlarge_transition_matrix()

        # set incoming transition weights
        if incoming_transition_weights is not None:
            # if a list was used, we have to check the length, but then it's easy superimposition
            if isinstance(incoming_transition_weights, list):
                if len(incoming_transition_weights) != old_n:
                    raise Exception("incoming_transition_weights list must have length equal to current number of old states")
                for i, weight in enumerate(incoming_transition_weights):
                    self.W_hh[i][old_n] = float(weight)

            # if a dict was used we have to match hidden state names
            elif isinstance(incoming_transition_weights, dict):
                for state_ref, weight in incoming_transition_weights.items():
                    i = self.hs_ref(state_ref)
                    if i == old_n:
                        continue
                    self.W_hh[i][old_n] = float(weight)

        # set outgoing transition weights, same idea
        if outgoing_transition_weights is not None:
            if isinstance(outgoing_transition_weights, list):
                if len(outgoing_transition_weights) != old_n:
                    raise Exception("outgoing_transition_weights list must have length equal to current number of old states")
                for j, weight in enumerate(outgoing_transition_weights):
                    self.W_hh[old_n][j] = float(weight)

            elif isinstance(outgoing_transition_weights, dict):
                for state_ref, weight in outgoing_transition_weights.items():
                    j = self.hs_ref(state_ref)
                    if j == old_n:
                        continue
                    self.W_hh[old_n][j] = float(weight)

        # set self-transition
        self.W_hh[old_n][old_n] = float(self_transition_weight)

        # everything's changed, clear derived
        self.clear_derived()

    def replace_hidden_state(
        self,
        hs_ref: int | str,
        new_hidden_state: HiddenState,
        mode: str = "strict"
        ) -> None:
        # replace existing HiddenState in the model with another HiddenState while preserving Whh shape

        idx = self.hs_ref(hs_ref)

        if new_hidden_state.hidden_state_name != self.hidden_states[idx].hidden_state_name:
            # check uniqueness unless same name as the one it's replacing
            self.check_novel_hs_name(new_hidden_state.hidden_state_name)

        if mode == "strict":
            self.validate_hs_against_schema(new_hidden_state)
        elif mode == "force":
            self.force_fill_hs_emissions(new_hidden_state, missing_fill="zeros")
            self.validate_hs_against_schema(new_hidden_state)
        else:
            raise Exception("mode must be 'strict' or 'force'")

        self.hidden_states[idx] = new_hidden_state
        self.rebuild_lookup_objects()
        self.clear_derived()

    def replace_transition_weight(self, prev_state_ref: int | str, current_state_ref: int | str, new_weight: float) -> None:

        # replace one transition weight for i and j in W_hh

        i = self.hs_ref(prev_state_ref)
        j = self.hs_ref(current_state_ref)
        self.W_hh[i][j] = float(new_weight)
    
        self.clear_derived()

    def replace_transition_row(self, prev_state_ref: int | str, new_row: list[float]) -> None:
        # replace all outgoing transitions from one state

        i = self.hs_ref(prev_state_ref)
        self.W_hh[i] = [float(w) for w in new_row]

        self.clear_derived()

    def replace_transition_column(self, current_state_ref: int | str, new_column: list[float]) -> None:
        # replace all incoming transitions to one state

        j = self.hs_ref(current_state_ref)
        for i, weight in enumerate(new_column):
            self.W_hh[i][j] = float(weight)

        self.clear_derived()

    def replace_init_weight(self, hs_ref: int | str, new_weight: float) -> None:
        # replace init weight for one state

        hidden_state = self.get_hs(hs_ref)
        hidden_state.init_weight = float(new_weight)
        self.clear_derived()

    def build_W_eh(self) -> dict[str, dict[str, list[float]]]:
        # build nested dictionary of emission weights

        W_eh = {}

        for es in self.emission_sets:
            set_name = es.set_name
            W_eh[set_name] = {}
    
            for hidden_state in self.hidden_states:
                state_name = hidden_state.hidden_state_name
                W_eh[set_name][state_name] = list(hidden_state.emission_weights[set_name])

        return W_eh

    def normalize_weights_vector(self, weights: list[float]) -> list[float]:
        # normalize one vector of weights to probabilities

        total = sum(weights)

        if total == 0:
            return [0.0] * len(weights)
        return [w / total for w in weights]

    def build_P_init(self) -> dict[str, float]:
        # build normalized init probabilities
        weights = [hidden_state.init_weight for hidden_state in self.hidden_states]
        probs = self.normalize_weights_vector(weights)

        return {
            hidden_state.hidden_state_name: probs[i]
            for i, hidden_state in enumerate(self.hidden_states)
            }

    def build_P_hh(self) -> dict[str, dict[str, float]]:
        # build normalized transition probabilities
        P_hh = {}

        for i, prev_state in enumerate(self.hidden_states):
            row_probs = self.normalize_weights_vector(self.W_hh[i])
            P_hh[prev_state.hidden_state_name] = {}

            for j, current_state in enumerate(self.hidden_states):
                P_hh[prev_state.hidden_state_name][current_state.hidden_state_name] = row_probs[j]

        return P_hh

    def build_P_eh(self) -> dict[str, dict[str, list[float]]]:
        # build normalized emission probabilities
        P_eh = {}

        for emission_set in self.emission_sets:
            set_name = emission_set.set_name
            P_eh[set_name] = {}

            for hidden_state in self.hidden_states:
                state_name = hidden_state.hidden_state_name
                weights = hidden_state.emission_weights[set_name]
                P_eh[set_name][state_name] = self.normalize_weights_vector(weights)

        return P_eh

    def normalize_all(self) -> None:
        # build and store all derived normalized objects
        self.W_eh = self.build_W_eh()
        self.P_init = self.build_P_init()
        self.P_hh = self.build_P_hh()
        self.P_eh = self.build_P_eh()

    def validate_model(self) -> None:
        # check basic model consistency -

        # W_hh is a square 
        if len(self.W_hh) != len(self.hidden_states):
            raise Exception("W_hh must have one row per hidden state")
        for row in self.W_hh:
            if len(row) != len(self.hidden_states):
                raise Exception("each W_hh row must have one column per hidden state")

        # etc etc
        return