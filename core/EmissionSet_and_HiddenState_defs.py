from typing import Optional

class EmissionSet:

    ##### CLASS BASICS #####

    def __init__(
        self,
        name: Optional[str] = None,
        length: int = 0,
        value_names: Optional[list[str]] = None,
        default_weights: Optional[list[float]] = None
        ) -> None:
    
        self.set_name = name

        if length < 0:
            raise Exception("length must be >= 0")
        self.length = length

        if value_names is None:
            value_names = [f"value_{i}" for i in range(length)]
        else:
            if len(value_names) != length:
                raise Exception("len(value_names) must equal length")
        self.value_names = list(value_names)

        if default_weights is None:
            default_weights = [0.0] * length
        else:
            if len(default_weights) != length:
                raise Exception("len(default_weights) must equal length")
            default_weights = [float(w) for w in default_weights]
        self.default_weights = list(default_weights)

    def __repr__(self) -> str:
        # represents itself by its init definition
        return (
            f"EmissionSet(name={self.set_name},"
            f"length={self.length}, "
            f"value_names={self.value_names}, "
            f"default_weights={self.default_weights})"
        )

    def copy(self) -> "EmissionSet":
        # make a copy
        return EmissionSet(
            name=self.set_name,
            length=self.length,
            value_names=list(self.value_names),
            default_weights=list(self.default_weights)
            )
    
    ##### UPDATING STUFF #####

    def set_name_value(self, name: Optional[str]) -> None:
        # replace the set's name
        self.set_name = name

    def set_value_names(self, value_names: list[str]) -> None:
        # replace full value_names list
        if len(value_names) != self.length:
            raise Exception("new value_names must have length equal to self.length")
        self.value_names = value_names

    def set_default_weights(self, default_weights: list[float]) -> None:
        # replace full default_weights vector
        if len(default_weights) != self.length:
            raise Exception("new default_weights must have length equal to self.length")
        self.default_weights = default_weights

    def replace_value_name(self, index: int, new_name: str) -> None:
        # replace the value name at index
        self.value_names[index] = new_name

    def replace_default_weight(self, index: int, new_weight: float) -> None:
        # replace the default weight at index
        self.default_weights[index] = new_weight

    def add_emission_value(self, value_name: str, default_weight: float = 0.0) -> None:
        # append a new emission value (increasing length by 1)
        self.value_names.append(value_name)
        self.default_weights.append(float(default_weight))
        self.length += 1


class HiddenState:

    ##### CLASS BASICS #####

    def __init__(
        self,
        name: Optional[str] = None,
        init_weight: float = 0.0,
        emission_weights: Optional[dict[str, list[float]]] = None
        ) -> None:
    
        self.hidden_state_name = name
        self.init_weight = float(init_weight)

        if emission_weights is None:
            emission_weights = {}

        self.emission_weights = {
            str(set_name): [float(w) for w in weights]
            for set_name, weights in emission_weights.items()
            }

    def __repr__(self) -> str:
        # represents itself by its init definition
        return (
            f"HiddenState(name={self.hidden_state_name}, "
            f"init_weight={self.init_weight}, "
            f"emission_weights={self.emission_weights})"
        )

    def copy(self) -> "HiddenState":
        # make a copy
        return HiddenState(
            name=self.hidden_state_name,
            init_weight=self.init_weight,
            emission_weights={k: list(v) for k, v in self.emission_weights.items()}
            )
    
    ##### UPDATING STUFF #####
    
    def set_name_value(self, name: Optional[str]) -> None:
        # replace the hidden state's name
        self.hidden_state_name = name

    def set_init_weight(self, weight: float) -> None:
        # replace the init weight
        self.init_weight = float(weight)

    def set_emission_weights(self, emission_set_name: str, weights: list[float]) -> None:
        # replace full weight vector for one emission set
        self.emission_weights[emission_set_name] = [float(w) for w in weights]

    def replace_emission_weight(self, emission_set_name: str, value_index: int,new_weight: float) -> None:
        # replace one emission weight within one emission set
        if emission_set_name not in self.emission_weights:
            raise Exception(f"emission set {emission_set_name} not present in state")
        self.emission_weights[emission_set_name][value_index] = float(new_weight)

