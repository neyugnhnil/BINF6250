# /core

`/core` contains code shared for all three HMM projects.

## Object Classes

Object classes were defined to store HMM models in an extensible way, supporting incremental model construction and updates.

___________________

### `EmissionSet`
- Represents one category of possible observed values, where only one value from that category can be observed at a time.
- Defines one part of the model’s observation schema.
- Stores:
    - An optional name.
    - Length of the emission set.
    - A name for each value.
    - A default weight for each value.
- Its `default_weights` may be used by `HMModel` as fallback emission weights when hidden states are filled or extended.
- Does not itself enforce model-wide name uniqueness; that is checked when the set is added to an `HMModel`.
- The order of `value_names` matters, since matching emission weight vectors are interpreted by index.
- Changing the number or order of values changes the expected structure of corresponding hidden state emission weights.

#### Initialization
`EmissionSet(name=None, length=0, value_names=None, default_weights=None)`
- Creates a new emission set.
- If `value_names` is omitted, placeholder names are created automatically.
- If `default_weights` is omitted, all defaults are set to `0.0`.
- `length` must be non-negative.

#### Methods
- `copy()`
  - Returns an independent copy of the emission set.
- `set_name_value(name)`
  - Replaces the set name.
- `set_value_names(value_names)`
  - Replaces the full list of value names.
  - The new list must match the current `length`.
- `set_default_weights(default_weights)`
  - Replaces the full list of default weights.
  - The new list must match the current `length`.
- `replace_value_name(index, new_name)`
  - Replaces one value name by index.
- `replace_default_weight(index, new_weight)`
  - Replaces one default weight by index.
- `add_emission_value(value_name, default_weight=0.0)`
  - Appends a new emission value and its default weight.
  - Increases `length` by 1.

--------------------------

### `HiddenState`

- Represents one hidden state in the HMM.
- Stores the state-specific parameters associated with that hidden state.
- Stores:
    - An optional name.
    - An initial weight.
    - A dictionary of emission weight vectors keyed by emission set name.
- Does not itself enforce agreement with a model’s emission schema; that is checked by `HMModel`.
- emission weight vectors are interpreted according to the order of values in the corresponding `EmissionSet`.

#### Initialization

`HiddenState(name=None, init_weight=0.0, emission_weights=None)`
- Creates a new hidden state.
- If `emission_weights` is omitted, the state starts with an empty dictionary.
- `init_weight` is stored as a float.
- Provided emission weights are stored as numeric lists keyed by emission set name.

#### Methods

- `copy()`
  - Returns an independent copy of the hidden state.
- `set_name_value(name)`
  - Replaces the hidden state name.
- `set_init_weight(weight)`
  - Replaces the initial weight.
- `set_emission_weights(emission_set_name, weights)`
  - Replaces the full emission weight vector for one emission set.
- `replace_emission_weight(emission_set_name, value_index, new_weight)`
  - Replaces one emission weight within one emission set.
  - Raises an error if that emission set is not present in the state.

--------------------------

### `HMModel`

- Represents the full hidden Markov model.
- Stores the model-wide emission schema, hidden states, and hidden-to-hidden transition weights.
- Enforces consistency between hidden states and the model’s emission sets.
- Stores:
    - `emission_sets`, the list of legal `EmissionSet` objects.
    - `hidden_states`, the list of `HiddenState` objects.
    - `W_hh`, the hidden-to-hidden transition weight matrix.
    - `es_lookup`, a cached emission set-name to index lookup dictionary.
    - `hs_lookup`, a cached hidden state-name to index lookup dictionary.
    - Derived objects built only when requested:
        - `W_eh`, a nested emission weight lookup dictionary
        - `P_init`, normalized initial-state probabilities
        - `P_hh`, normalized transition probabilities
        - `P_eh`, normalized emission probabilities
- Treats the model’s emission sets as the schema that all hidden states must follow.
- Rebuilds cached lookup dictionaries whenever emission sets or hidden states are structurally changed.
- Clears derived probability objects whenever the model is changed.

#### Initialization

`HMModel(emission_sets=None, hidden_states=None)`
- Creates a new HMM object.
- Initializes empty emission set, hidden state, and transition weight structures.
- Initializes empty cached lookup dictionaries.
- Adds provided emission sets first, then provided hidden states.
- Hidden states added at initialization are checked in `"strict"` mode.
- Derived probability objects are not built until requested.

#### Methods

##### Internal helpers
- `rebuild_lookup_objects()`
  - Rebuilds the cached name-to-index lookup dictionaries for emission sets and hidden states.
  - Used internally after structural changes to avoid repeated linear scanning during reference resolution.
- `es_ref(es_ref)`
  - Normalizes an emission set reference.
  - Accepts either an integer index or an emission set name.
  - If a name is provided, resolves it through `es_lookup`.
- `hs_ref(hs_ref)`
  - Normalizes a hidden state reference.
  - Accepts either an integer index or a hidden state name.
  - If a name is provided, resolves it through `hs_lookup`.
- `get_es(es_ref)`
  - Returns an `EmissionSet` object by reference.
- `get_hs(hs_ref)`
  - Returns a `HiddenState` object by reference.
- `clear_derived()`
  - Clears all derived lookup and probability objects.
- `check_novel_es_name(name)`
  - Raises an error if an emission set name is already present in the model.
- `check_novel_hs_name(name)`
  - Raises an error if a hidden state name is already present in the model.
- `validate_hs_against_schema(hidden_state)`
  - Checks that a hidden state matches the model’s emission schema exactly.
  - Requires the same emission set names and the correct weight-vector length for each set.
- `force_fill_hs_emissions(hidden_state, missing_fill="zeros")`
  - Used internally in `"force"` mode.
  - Rejects emission sets not already present in the model.
  - Fills any missing emission sets with zero weights or with that emission set’s `default_weights`.
- `enlarge_transition_matrix()`
  - Expands `W_hh` by one row and one column.
  - Returns the previous number of hidden states.

#### External methods for modifying the model
- `copy()`
  - Returns an independent copy of the full model, including stored and derived objects.
- `add_emission_set(new_emission_set, fill_hidden_states_with="zeros")`
  - Adds a new emission set to the model schema.
  - Rebuilds cached lookup objects.
  - Extends existing hidden states if they do not yet contain that emission set.
  - Missing emission weights are filled with zeros or with the emission set’s `default_weights`.
- `add_hidden_state(new_hidden_state, mode="strict", missing_fill="zeros", incoming_transition_weights=None, outgoing_transition_weights=None, self_transition_weight=0.0, update_init_weight=None)`
  - Adds a new hidden state to the model.
  - In "strict" mode, the state must already match the model’s emission schema exactly.
  - In "force" mode, missing emission sets may be inserted automatically.
  - Extra emission sets not already in the model cause rejection.
  - Option to replace the state’s initial weight during insertion.
  - Rebuilds cached lookup objects.
  - Expands the transition matrix by one row and one column.
  - Optional incoming, outgoing, and self-transition weights may be set during addition.
- `replace_hidden_state(hs_ref, new_hidden_state, mode="strict")`
  - Replaces one hidden state while preserving transition-matrix shape.
  - Applies the same schema rules as hidden state addition.
  - Rebuilds cached lookup objects afterward.
- `replace_transition_weight(prev_state_ref, current_state_ref, new_weight)`
  - Replaces one transition weight.
- `replace_transition_row(prev_state_ref, new_row)`
  - Replaces all outgoing transition weights from one state.
- `replace_transition_column(current_state_ref, new_column)`
  - Replaces all incoming transition weights to one state.
- `replace_init_weight(hs_ref, new_weight)`
  - Replaces one state’s initial weight.

#### External methods for deriving lookup/probability objects
- `build_W_eh()`
  - Builds the nested emission weight lookup dictionary.
- `normalize_weights_vector(weights)`
  - Normalizes one weight vector into probabilities.
  - If the total weight is zero, returns a zeroes vector of the same length.
- `build_P_init()`
  - Builds normalized initial-state probabilities.
- `build_P_hh()`
  - Builds normalized transition probabilities.
- `build_P_eh()`
  - Builds normalized emission probabilities.
- `normalize_all()`
  - Builds and stores all derived lookup and probability objects.
- `validate_model()`
  - Performs structural consistency checks.