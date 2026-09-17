"""Six instances of one connectome, batched as six columns.

The central design claim: six instances cost near-nothing over one, and
bit-identity between them is STRUCTURAL rather than something we tune for.

Both properties come from the same decision. State is (N, 6). Every step does
one `csr @ state` with a fixed reduction order, so all six columns are reduced
in the same order by the same code path. There is no per-instance kernel, no
atomic accumulation, and no spike-order dependence anywhere.

We deliberately use the deterministic pull model even though it reads the whole
matrix each step. project.md says "event-driven propagation is not optional".
That was written for a realtime target. A push kernel with atomics
accumulates in non-deterministic order, so the six would decorrelate from
floating-point noise instead of from stimulation. Offline we do not need the
speed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from ..config import LIF, N_INSTANCES

log = logging.getLogger(__name__)


@dataclass
class State:
    """(N, n_instances) column-batched state. ~4MB at N=166,700, 6 instances."""

    v: np.ndarray  # membrane potential, mV
    g: np.ndarray  # synaptic conductance
    refractory: np.ndarray  # steps remaining
    delay_buf: np.ndarray  # (delay_steps, N, n_instances) spike ring buffer
    step: int = 0

    @classmethod
    def initial(cls, n: int, lif: LIF, n_instances: int = N_INSTANCES) -> "State":
        # Every instance starts at exactly the same potential. At t=0 the six
        # are the same program in the same state; only what is done to them
        # differs.
        return cls(
            v=np.full((n, n_instances), lif.v_reset_mv, dtype=np.float32),
            g=np.zeros((n, n_instances), dtype=np.float32),
            refractory=np.zeros((n, n_instances), dtype=np.int32),
            delay_buf=np.zeros(
                (lif.delay_steps, n, n_instances), dtype=np.float32
            ),
        )


class Engine:
    """Deterministic batched LIF over one shared CSR."""

    def __init__(self, csr: sp.csr_matrix, lif: LIF | None = None,
                 n_instances: int = N_INSTANCES):
        self.lif = lif or LIF()
        self.csr = csr.astype(np.float32)
        self.csr.sort_indices()  # fixed per-row order == deterministic reduction
        self.n = csr.shape[0]
        self.n_instances = n_instances

        # Precomputed decay factors. Computed once so every step uses bitwise
        # identical constants.
        self.decay_v = np.float32(np.exp(-self.lif.dt_ms / self.lif.tau_membrane_ms))
        self.decay_g = np.float32(np.exp(-self.lif.dt_ms / self.lif.tau_syn_ms))
        self.w_syn = np.float32(self.lif.weight_per_synapse_mv)
        self.v_th = np.float32(self.lif.v_threshold_mv)
        self.v_rst = np.float32(self.lif.v_reset_mv)

    def step(self, state: State, injection: np.ndarray | None = None) -> np.ndarray:
        """Advance one dt. Returns (N, n_instances) bool spike matrix.

        `injection` is (N, n_instances) added current in mV. This is the ONLY
        way instances can differ, so isolation follows from the structure.
        """
        lif = self.lif

        # 1. Spikes emitted `delay_steps` ago arrive now.
        slot = state.step % lif.delay_steps
        arriving = state.delay_buf[slot]

        # 2. Gather. One matmul, all six columns, fixed CSR index order.
        #    scipy's csr_matmat is a sequential per-row accumulation, so the
        #    reduction order is fixed by `indices` and is identical per column.
        current = self.csr @ arriving

        # 3. Conductance decays, then takes this step's input.
        state.g *= self.decay_g
        state.g += current * self.w_syn

        # 4. Membrane decays toward reset, then integrates.
        state.v -= self.v_rst
        state.v *= self.decay_v
        state.v += self.v_rst
        state.v += state.g

        if injection is not None:
            state.v += injection

        # 5. Refractory neurons are clamped, not integrated.
        in_refractory = state.refractory > 0
        state.v[in_refractory] = self.v_rst
        state.refractory[in_refractory] -= 1

        # 6. Threshold.
        spikes = state.v >= self.v_th
        state.v[spikes] = self.v_rst
        state.refractory[spikes] = lif.refractory_steps

        # 7. Publish into the ring buffer for arrival `delay_steps` from now.
        state.delay_buf[slot] = spikes.astype(np.float32)

        state.step += 1
        return spikes


def population_rate(spikes: np.ndarray) -> np.ndarray:
    """(n_instances,) spike count. The always-on summary the plots consume."""
    return spikes.sum(axis=0)
