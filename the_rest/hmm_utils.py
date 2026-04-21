# Extensible Hidden Markov Model Module for use in projects 8-10

# import statements


class Emission_Set:
    """
    Represents one named set of mutually exclusive possible observed values
    This object represents the emissions and associated emission probabilities for one state
    in a Hidden Markov Model

    Attributes:
        set_name: optional (recommended) identifier for the emission set. 
                  If used, no two emission sets within the same HMM should share a name
        length: number of emission values in the set. Always equal to the number of value weights. 
        value_names: list of strings naming the emissions in the set
        weights: list of numeric weights (floats), used as fallback weights when a hidden state
                         does not yet have a specific weight vector for this emission set
                         NOTE: weights are used in place of probabilities until the instant a probability needs to be rolled

    Methods:
        rename: renames the emission set
        replace_all_value_names: renames ALL emissions in the set
        replace_one_value_name: renames ONE emission in the set
        replace_default_weights: updates ALL weights in self.default_weights to new values
        replace_one_weight: updates ONE weight at the given position in self.default_weights
        add_emission_value: adds a new emission to the set, requiring a name for the new emission at a minimum
    """
    def __init__(self, length: int = 1, name: str | None = None, 
                 value_names: list[str] | None = None,
                 weights: list[float] | None = None):
        # define the number of emissions in this set
        self.length = length

        # removing the spaces in the name just so it's visually clear during debugging where the name ends
        self.name = name.replace(" ", "_") if name is not None else "Unnamed_emission_set"

        # the comprehension is doing all that so all the names are strings to avoid breaking later methods
        self.value_names = value_names if value_names is not None else list(f"emission_{i}" for i in range(length))

        self.default_weights = weights if weights is not None else [0.0] * self.length
    
    def rename(self, new_name: str):
        """
        Function to rename the emission set

        @param new_name: string, the new name for this emission set
        @return: nothing, just updates attributes
        """
        # removing the spaces in the name just so it's visually clear during debugging where the name ends
        self.name = new_name.replace(" ", "_")
    
    def replace_all_value_names(self, new_names: list[str]):
        """
        Function to assign new names to ALL of the emisisons in this emission set
        
        @param new_names: list[str], expected to have length equal to this set's length
                          these are the new names for all of the emissions in the set, preserving
                          the order in which the emissions were listed previously
        @return: nothing, just updates attributes
        """
        # check if new_names is of appropriate length
        if len(new_names) == self.length:
            
            # reassign the names for the values
            self.value_names = new_names

        else:
            # print for debugging
            print(f"List of new emission names is too short, length is {len(new_names)}, "
                  f"expected length to be {self.length}.\n"
                  f"List at issue is: {new_names}")
    
    def replace_one_value_name(self, old_name: str, new_name: str):
        """
        Function to replace the name for ONE emission in this set
        
        @param old_name: str, indicates the existing name that needs to be replaced
        @param new_name: str, indicates the name that this emission is being reassigned to
        @return: nothing, just updates attributes
        """
        # check if old_name is actually in our list of names
        if old_name in self.value_names:

            # find the name's position in the current list of emission names
            idx_to_fix = self.value_names.index(old_name)

            # reassign the emission at this position to the new name
            self.value_names[idx_to_fix] = new_name

        else:
            # print for debugging, otherwise do literally nothing
            print(f"Attempted to change an emission's name from {old_name} to {new_name}, but FAILED "
                  f"because {old_name} is not an emission in {self.name}. \n"
                  "Current emission names are: {self.value_names})")
    
    def replace_default_weights(self, new_weights: list[float]):
        """
        Function to replace the full default weights vector, assuming emission order is intended to be preserved
        
        @param new_weights: list of floats representing the new default weights for the emissions
        @return: nothing, just updates attributes
        """
        # check if new_weights is of appropriate length
        if len(new_weights) == self.length:

            # update the default weights directly
            self.default_weights = new_weights

        else:

            # print for debugging
            print(f"Attempted to update default weights for Emission_Set {self.name}, but failed "
                  f"because the list of new weights is of length {len(new_weights)}, expceted {self.length}.")
        
    def replace_one_weight(self, position_to_replace: int, new_weight: float):
        """
        Function to replace ONE of the values in the default weights list
        
        @param position_to_replace: int, represents the list index for the weight to be replaced
        @param new_weight: float, the new value for the weight of interest
        @return: nothing, just updates object attributes
        """
        # check that position_to_replace won't give us an index error
        try:
            self.default_weights[position_to_replace] = new_weight
        except IndexError as err:
            print(f"Encountered index error while running {self.name}.replace_one_weight(): {err}")
        
    def add_emission_value(self, value_name: str, default_weight: float = 0.0):
        """
        Function to add a new emission to this emission set (updates list of names, list of weights, and the length)
        
        @param name: str, the name for this emission
        @param default_weight: float, the default weight for this emission
        @return: nothing, just updates attributes
        """
        self.value_names.append(value_name)
        self.default_weights.append(default_weight)
        self.length += 1
    
    def __repr__(self) -> str:
        """
        Function to print out the string representation of this emission set
        """

        return f"{self.name} has {self.length} emissions:\n{self.value_names}\nwith weights:\n{self.default_weights}"


class Hidden_State:
    """
    Class to represent one hidden state in a Hidden Markov Model. NOT USABLE IN ISOLATION
    
    Attributes:
        Name for this hidden state
        Initial weight: the probability of the model traversing to this state from the initial state
        emission weights: dictionary that maps the name of an emission set to the weights of the emissions in that set
    Methods:
        rename: renames the hidden state
        change_init_weight: changes the weight for this hidden state being transitioned to FROM the initial state of an HMM
        replace_emission_weights: changes the full list of emission weights for a named set of emissions
        replace_emission_weight: changes ONE emission in a named set of emissions

    """

    def __init__(self, name: str = "Unnamed_state",
                 init_weight: float = 0.0,
                 emission_set: Emission_Set | None = None):
        self.name = name

        self.init_weight = init_weight

        # initialize the dict mapping emission set names to their emissions' weights
        self.emission_weights = {}

        # check if an emission set was even passed
        if emission_set is not None:
            
            # since an emission set was passed, 
            self.emission_weights[emission_set.name] = emission_set.default_weights
        
        else:
            self.emission_weights["no_emission_set"] = [0.0]
        
    def rename(self, new_name: str):
        """
        Function to rename the hidden state
        
        @param new_name: str, the new name for this hidden state
        @return: nothing, just updates attributes
        """

        self.name = new_name
    
    def change_init_weight(self, new_weight: float):
        """
        Function to change the probability of transitioning to this hidden state from the initial state for an HMM
        NOTE: this adjusts the WEIGHT, NOT the PROBABILITY. Weights are used to calculate the probability of a transition,
        but weights are NOT probabilities, themselves
        
        @param new_weight: float, the new initial weight for this state
        @return: nothing, just updates attributes
        """
        self.init_weight = new_weight

    def replace_emission_weights(self, set_to_change: str, new_weights: list[float]):
        """
        Function to replace ALL emission weights for a given emission set
        
        @param set_to_change: str, the name of the emission set whose weights are being updated
        @param new_weights: list[float], the new weights for the emissions in the given set, MUST match
                            the shape of the emission set's existing list of weights
        @return: nothing, just updates attributes
        """
        # check if set_to_change is even in this hidden state
        try:
            new_shape = len(new_weights)
            old_shape = len(self.emission_weights[set_to_change])
            # check if new_weights has the same length as the list of weights we want to change
            if new_shape == old_shape:

                # update the full list of weights for this emission set
                self.emission_weights[set_to_change] = new_weights

            else:

                # fail to make the change and print for debugging
                print(f"Attempted to change emission weights for hidden state {self.name}'s emission set: {set_to_change}\n"
                      f"Operation failed because new list of weights had length {new_shape}, expected length {old_shape}.")
        
        except KeyError as err:
            # this should only run if set_to_change isn't in this state's emission sets
            print(f"Attempted to change {set_to_change}'s emission weights in {self.name}, caused KeyError:\n{err}")

    def replace_emission_weight(self, set_to_change: str, position: int, new_weight: float):
        """
        Function to update the weight for ONE emission in the named emission set
        
        @param set_to_change: str, the name for the emission where a weight is getting changed
        @param position: int, the index for the emission weight that needs to be changed
        @param new_weight: float, the new value for the weight that's being changed
        @return: nothing, just updates attributes
        """
        try:
            # attempt to access the list of emission weights for the named set
            curr_weights = self.emission_weights[set_to_change]

            # attempt to update the weight at the given position
            curr_weights[position] = new_weight
        except KeyError as err:
            print(f"Failed to update the emission for {self.name} in the set {set_to_change},\n{err}")
        
        except IndexError as err:
            print(f"Failed to update the {position}th emission under {set_to_change} for Hidden State {self.name}:\n{err}")


class Hidden_Markov_Model:
    """
    Class to represent a Hidden Markov Model

    Explicit description of a design choice we made:
        The model’s emission schema is defined by emission_sets, and all hidden states must conform to it. 
        Strict mode requires exact agreement. 
        Force mode fills missing emission weights; so it allows for adding models which LACK weights for 
        certain emission sets already in the schema. 
        Extra emission sets are always rejected unless added to the model first via add_emission_set.
    
    Attributes:
        emission_sets: list of emission_set objects for the model, defining allowed emission schema
        hidden_states_list: list of hidden_state objects for the model
        tw_matrix: maps the weight for a transition from prev_state to _curr_state
        ew_matrix: nested dict representation of emission weights
                                ew_matrix[emission_set][hidden_state][emission_value] = weight
                                Derived from hidden_states_list and emission_sets when requested
        ep_matrix: like emission_weight_matrix, but using probabilities, rather than weights
        tp_matrix: like transition_weight_matrix, but using probabilities, rather than weights
        init_probs: dict mapping hidden states to their inital probabilities, calculated using the states' initial weights
    Model-Building Methods:
        add_emission_set: adds an emission set object if it does not share a name with other emission sets in the model
        add_hidden_state: adds a new hidden state object if it doesn't share a name with another hidden state in the model
        _fill_missing_state_emissions: forcibly inserts a weight vector into a state when it lacks an emission set that is
                                        present in the model (primarily used during forced add_hidden_state)
    Matrix/dict-building Methods:
        build_emission_weight_matrix: builds the emission weight matrix tracking the weight for all emissions across all states
        _calculate_init_probs: creates a dictionary mapping state names to their initial probabilities, calculated using all
                              the states' initial weight values
        _build_tp_matrix: builds a matrix mapping the probability of a state transition from old_state to new_state
        _build_ep_matrix: builds a matrix tracking the probability of every emission across all states
        normalize_all: calls the three probability functions above and stores the results to object attributes
    Lookup Methods:
        get_emission_set: returns emission set object
        get_hidden_state: returns hidden state object
        get_emission_set_index: returns the named emission set's position in self.emission_sets
        get_hidden_state_index: returns the named hidden state's position in self.hidden_states
    Other Methods:
        __repr__: returns string representation of model
        
    """