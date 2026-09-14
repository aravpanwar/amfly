"""CUDA engine. Same arithmetic as the numpy engine, on the GPU.

The whole project rests on six instances being bit-identical under identical
input, so this backend is only worth having if it preserves that. Two settings
make it safe and both are mandatory:

  - `torch.use_deterministic_algorithms(True)`, so no kernel picks a
    non-deterministic reduction.
  - TF32 off. Ampere and later silently truncate fp32 mantissa bits in matmul
    by default, which is fast and fatal to a bit-identity claim.

Sparse CSR times dense (N, 6) is one kernel over all six columns with a fixed
reduction order, exactly as on CPU. No atomics, no scatter, no spike-order
dependence.

Note the GPU is NOT expected to match the CPU backend bit for bit. Different
hardware reduces in a different (but internally fixed) order. What must hold is
that the six instances match *each other* on whichever backend is running, and
that a rerun on the same machine reproduces itself. That is what the gates
check, and `docs/decisions.md` states the limit.
"""

from __future__ import annotations

import logging

import numpy as np
import scipy.sparse as sp

from ..config import LIF, N_INSTANCES

log = logging.getLogger(__name__)


class CudaEngine:
    """Batched LIF on the GPU. Mirrors Engine step for step."""

    def __init__(
        self,
        csr: sp.csr_matrix,
        lif: LIF | None = None,
        n_instances: int = N_INSTANCES,
        device: str = "cuda",
    ):
        import torch

        self.torch = torch
        self.lif = lif or LIF()
        self.n = csr.shape[0]
        self.n_instances = n_instances
        self.device = torch.device(device)

        csr = csr.astype(np.float32)
        csr.sort_indices()  # fixed per-row order, same guarantee as on CPU

        self.csr = torch.sparse_csr_tensor(
            torch.from_numpy(csr.indptr.astype(np.int64)),
            torch.from_numpy(csr.indices.astype(np.int64)),
            torch.from_numpy(csr.data),
            size=(self.n, self.n),
            dtype=torch.float32,
            device=self.device,
        )

        self.decay_v = float(np.exp(-self.lif.dt_ms / self.lif.tau_membrane_ms))
        self.decay_g = float(np.exp(-self.lif.dt_ms / self.lif.tau_syn_ms))
        self.w_syn = float(self.lif.weight_per_synapse_mv)
        self.v_th = float(self.lif.v_threshold_mv)
        self.v_rst = float(self.lif.v_reset_mv)

        log.info(
            "cuda engine: %d neurons, %d instances, %.0f MB graph",
            self.n, n_instances,
            (csr.data.nbytes + csr.indices.nbytes * 2 + csr.indptr.nbytes * 2) / 1e6,
        )

    def initial_state(self) -> dict:
        t = self.torch
        return {
            "v": t.full((self.n, self.n_instances), self.v_rst,
                        dtype=t.float32, device=self.device),
            "g": t.zeros((self.n, self.n_instances),
                         dtype=t.float32, device=self.device),
            "refractory": t.zeros((self.n, self.n_instances),
                                  dtype=t.int32, device=self.device),
            "delay_buf": t.zeros(
                (self.lif.delay_steps, self.n, self.n_instances),
                dtype=t.float32, device=self.device,
            ),
            "step": 0,
        }

    def step(self, state: dict, injection=None):
        """One dt. Returns a (N, n_instances) bool tensor on the device.

        Operation order is identical to Engine.step so the two backends stay
        comparable line for line.
        """
        t = self.torch
        lif = self.lif

        slot = state["step"] % lif.delay_steps
        arriving = state["delay_buf"][slot]

        current = t.sparse.mm(self.csr, arriving)

        state["g"] *= self.decay_g
        state["g"] += current * self.w_syn

        state["v"] -= self.v_rst
        state["v"] *= self.decay_v
        state["v"] += self.v_rst
        state["v"] += state["g"]

        if injection is not None:
            state["v"] += injection

        in_refractory = state["refractory"] > 0
        state["v"] = t.where(
            in_refractory,
            t.full_like(state["v"], self.v_rst),
            state["v"],
        )
        state["refractory"] = t.where(
            in_refractory, state["refractory"] - 1, state["refractory"]
        )

        spikes = state["v"] >= self.v_th
        state["v"] = t.where(
            spikes, t.full_like(state["v"], self.v_rst), state["v"]
        )
        state["refractory"] = t.where(
            spikes,
            t.full_like(state["refractory"], lif.refractory_steps),
            state["refractory"],
        )

        state["delay_buf"][slot] = spikes.to(t.float32)
        state["step"] += 1
        return spikes


def available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())
