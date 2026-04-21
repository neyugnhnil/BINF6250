# toy data for first order HMM implementation (downstream from class defs)
# usable on anything that assumes HMModel class

from core.EmissionSet_and_HiddenState_defs import EmissionSet, HiddenState
from core.HMModel_def import HMModel

# emission sets
dna_base = EmissionSet(
    name="dna_base",
    length=4,
    value_names=["A", "T", "C", "G"],
    default_weights=[1, 1, 1, 1]
    )

methylation = EmissionSet(
    name="methylation",
    length=2,
    value_names=["unmethylated", "methylated"],
    default_weights=[2, 1]
    )

# hidden states
background = HiddenState(
    name="background",
    init_weight=3.0,
    emission_weights={
        "dna_base": [2, 2, 1, 1],
        "methylation": [9, 1]
    }
)

CpG_island = HiddenState(
    name="CpG_island",
    init_weight=1.0,
    emission_weights={
        "dna_base": [1, 1, 5, 5],
        "methylation": [1, 9]
    }
)

# build model

model = HMModel(
    emission_sets=[dna_base, methylation],
    hidden_states=[background, CpG_island]
    )

# specify transition weights
model.replace_transition_row("background", [9, 1])
model.replace_transition_row("CpG_island", [1, 9])

# fake observations

emissions = [
    ["A", "unmethylated"],
    ["T", "unmethylated"],
    ["C", "methylated"],
    ["G", "methylated"],
    ["C", "methylated"],
    ["A", "unmethylated"],
    ["T", "unmethylated"],
    ["C", "unmethylated"],
    ["C", "unmethylated"],
    ["C", "methylated"],
    ["T", "methylated"],
    ["C", "methylated"],
    ["G", "methylated"],
    ["C", "methylated"],
]