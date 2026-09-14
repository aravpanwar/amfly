"""Every constant in one place, frozen.

Numbers here that describe the dataset were measured from the actual files, not
copied from anyone's README. See MEASURED below. If a number in this file ever
disagrees with the data, the data wins and the test suite should fail loudly.
"""

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

BASE_URL = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/"
    "connectome-data/flat-connectome/"
)

# Only these three. syn-points (12.7GB) and syn-partners (6.8GB) are per-synapse
# coordinates, which are for anatomy, not for LIF dynamics. We do not need them.
FILES = {
    "weights": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "neurotransmitters": "body-neurotransmitters-male-cns-v1.0.feather",
}


@dataclass(frozen=True)
class Measured:
    """Computed directly from the v1.0 files. Asserted in tests/test_loader.py.

    Note these are three genuinely different quantities that are routinely
    conflated in other projects. 25.6M is the *connection* count. The synapse
    count is 124M. Saying "25.6M synapses" is wrong.
    """

    neurons: int = 166_700
    connections: int = 25_582_938
    synapses: int = 124_177_617
    descending_neurons: int = 1_314

    # Raw, before filtering. The weights table is 151.8M rows: reading it
    # naively as int64 costs ~3.6GB of RAM. Project columns, filter, downcast.
    annotations_raw_rows: int = 211_577
    weights_raw_rows: int = 151_856_684


MEASURED = Measured()


# ---------------------------------------------------------------------------
# Neurotransmitter sign policy (Dale's law: sign is per presynaptic neuron)
# ---------------------------------------------------------------------------

# Glutamate is INHIBITORY in the fly (GluCl). This is the Shiu et al. convention
# and it surprises people coming from vertebrate work, so it is stated loudly
# here and in the README rather than left implicit.
EXCITATORY = ("acetylcholine", "dopamine", "octopamine", "serotonin")
INHIBITORY = ("gaba", "glutamate", "histamine")

# consensus_nt of "unclear", or no row at all. Measured: 2,999 + 178 = 3,177
# neurons, 1.9% of the network. Defaulting these to excitatory is a judgement
# call, so it is declared, counted at load time, and shipped with a sensitivity
# run that silences them instead.
UNCLEAR_SIGN = 1.0

# Measured distribution of consensus_nt over the 166,700 real neurons.
NT_COUNTS = {
    "acetylcholine": 103_720,
    "glutamate": 29_302,
    "gaba": 22_069,
    "histamine": 7_891,
    "unclear": 2_999,
    "dopamine": 392,
    "__missing__": 178,
    "octopamine": 101,
    "serotonin": 48,
}


# ---------------------------------------------------------------------------
# LIF parameters, Shiu et al. 2024 (Nature)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LIF:
    """Defaults from philshiu/Drosophila_brain_model.

    Caveat that belongs in the README: Shiu et al. is brain-only, ~127K neurons.
    We run whole-CNS at 166,700 including the nerve cord. That extension is ours,
    not theirs, and we say so rather than implying they validated it.
    """

    tau_membrane_ms: float = 20.0
    v_threshold_mv: float = -45.0
    v_reset_mv: float = -52.0
    refractory_ms: float = 2.2
    weight_per_synapse_mv: float = 0.275
    tau_syn_ms: float = 5.0
    delay_ms: float = 1.8

    # 0.1ms is the Brian2 default clock, NOT a published parameter of the paper.
    # project.md cited it as the latter. Cite it honestly as our integration choice.
    dt_ms: float = 0.1

    @property
    def delay_steps(self) -> int:
        return int(round(self.delay_ms / self.dt_ms))

    @property
    def refractory_steps(self) -> int:
        return int(round(self.refractory_ms / self.dt_ms))


# ---------------------------------------------------------------------------
# The piece
# ---------------------------------------------------------------------------

N_INSTANCES = 6
OPERATOR = 5  # index of the fly at the dial; 0..4 are the chambers
CHAMBERS = (0, 1, 2, 3, 4)

# Heat targets. There are ZERO neurons of type "AC" in this dataset, despite the
# thermosensory literature naming them, so an AC lookup silently returns nothing.
# Match these exact prefixes.
#
# Never match "contains VP": LoVP* is a large optic-lobe visual family that would
# poison the heat channel invisibly.
THERMO_PREFIX = "TRN_VP"  # 4 types, 25 bodies. Primary heat channel.
HYGRO_PREFIX = "HRN_VP"  # 4 types, 66 bodies. Secondary, off by default.

# Perceptible lag between an operator spike and the chamber responding, so a
# viewer can learn to predict it before understanding why.
DIAL_LATENCY_MS: float = 500.0

SUPERCLASS_DESCENDING = "descending_neuron"


@dataclass(frozen=True)
class Config:
    lif: LIF = field(default_factory=LIF)
    n_instances: int = N_INSTANCES
    weight_threshold: int = 1  # keep all 25.58M edges; >=2 is an opt-in fallback
    include_hygro: bool = False
    silence_unclear_nt: bool = False  # the sensitivity run
    seed_note: str = "no RNG in the divergence path; see README"
