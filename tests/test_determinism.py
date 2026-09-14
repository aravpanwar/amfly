"""Gates 1, 2, 3. The tests the piece actually depends on.

If Gate 1 fails, the six instances are decorrelating from floating-point noise
and every plot downstream is a fake of the real phenomenon.

If Gate 2 fails, we have built the fly_brain_food anti-pattern: one circuit
wearing six costumes. That shortcut is invisible in a demo and fatal here.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import scipy.sparse as sp

from amfly.config import LIF
from amfly.sim.engine import Engine, State


def _toy_csr(n: int = 400, seed: int = 0) -> sp.csr_matrix:
    """Small random graph. The seed builds the GRAPH, never the dynamics.

    There is no RNG anywhere in the divergence path; this only stands in for
    the real connectome so the gates run in milliseconds.
    """
    rng = np.random.default_rng(seed)
    density = 0.02
    m = sp.random(n, n, density=density, format="csr", dtype=np.float32,
                  random_state=rng)
    m.data = (rng.integers(1, 6, size=m.nnz) *
              rng.choice([1.0, -1.0], size=m.nnz)).astype(np.float32)
    m.sort_indices()
    return m


def _digest(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr)).hexdigest()


def _run(csr, steps, injection_fn=None, n_instances=6):
    lif = LIF()
    eng = Engine(csr, lif, n_instances=n_instances)
    st = State.initial(csr.shape[0], lif, n_instances)
    rec = np.zeros((steps, csr.shape[0], n_instances), dtype=bool)
    for t in range(steps):
        inj = injection_fn(t, csr.shape[0], n_instances) if injection_fn else None
        rec[t] = eng.step(st, inj)
    return rec


# --------------------------------------------------------------------------
# Gate 1: identical input -> bit-identical spike trains
# --------------------------------------------------------------------------

def test_gate1_instances_bit_identical_under_identical_input():
    csr = _toy_csr()
    steps = 300

    # A uniform kick applied equally to all six. Nothing distinguishes them.
    def uniform(t, n, k):
        inj = np.zeros((n, k), dtype=np.float32)
        if t < 50:
            inj[:20, :] = 3.0
        return inj

    rec = _run(csr, steps, uniform)

    ref = rec[:, :, 0]
    for i in range(1, 6):
        assert np.array_equal(ref, rec[:, :, i]), (
            f"instance {i} diverged from instance 0 under identical input. "
            "This is a bug, not a piece."
        )

    # Hash comparison too: array_equal on an all-false array would pass
    # vacuously, so assert the run is actually alive.
    digests = {_digest(rec[:, :, i]) for i in range(6)}
    assert len(digests) == 1
    assert rec.sum() > 0, "network silent; gate passed vacuously"


# --------------------------------------------------------------------------
# Gate 2: isolation. Heating chamber 3 must not touch 0, 1, 2, 4.
# --------------------------------------------------------------------------

def test_gate2_injection_isolates_to_one_chamber():
    csr = _toy_csr()
    steps = 300
    target = 3

    def heat_one(t, n, k):
        inj = np.zeros((n, k), dtype=np.float32)
        if t < 50:
            inj[:20, :] = 3.0          # baseline, all instances
        if 60 <= t < 200:
            inj[:10, target] += 4.0    # heat, chamber 3 only
        return inj

    def control(t, n, k):
        inj = np.zeros((n, k), dtype=np.float32)
        if t < 50:
            inj[:20, :] = 3.0
        return inj

    heated = _run(csr, steps, heat_one)
    base = _run(csr, steps, control)

    for i in (0, 1, 2, 4):
        assert np.array_equal(heated[:, :, i], base[:, :, i]), (
            f"chamber {i} changed when only chamber {target} was heated. "
            "One circuit is being shared across instances."
        )

    assert not np.array_equal(heated[:, :, target], base[:, :, target]), (
        "heating chamber 3 changed nothing; injection is not reaching the network"
    )


# --------------------------------------------------------------------------
# Gate 3: reproducibility across runs
# --------------------------------------------------------------------------

def test_gate3_rerun_is_byte_identical():
    csr = _toy_csr()

    def kick(t, n, k):
        inj = np.zeros((n, k), dtype=np.float32)
        if t < 40:
            inj[:20, :] = 3.0
        return inj

    a = _run(csr, 200, kick)
    b = _run(csr, 200, kick)
    assert _digest(a) == _digest(b)


def test_gate3_csr_index_order_is_canonical():
    """Determinism depends on sorted CSR indices; assert we never ship unsorted."""
    csr = _toy_csr()
    eng = Engine(csr)
    assert eng.csr.has_sorted_indices


# --------------------------------------------------------------------------
# The reason we do not use an atomic push kernel, as an executable note.
# --------------------------------------------------------------------------

def test_float_addition_is_not_associative():
    """Documents the trap: same values, different order, different answer.

    An event-driven push kernel with atomics accumulates in non-deterministic
    order. The gap below is ~20% of one 0.275mV synaptic weight, injected every
    step, which is more than enough to decorrelate six instances on its own.
    """
    rng = np.random.default_rng(0)
    vals = rng.random(10_000).astype(np.float32)
    forward = np.float32(0)
    for v in vals:
        forward += v
    backward = np.float32(0)
    for v in vals[::-1]:
        backward += v
    assert forward != backward


# --------------------------------------------------------------------------
# Synaptic delay. The ring buffer is easy to get off by one, and an error
# here would change every timing number in the project without failing
# anything else.
# --------------------------------------------------------------------------

def test_synaptic_delay_is_exactly_delay_steps():
    """A spike at t arrives at t + delay_steps, not t+1 and not t+delay-1."""
    lif = LIF()
    n = 2
    # neuron 0 -> neuron 1, 100 synapses.
    m = sp.csr_matrix(np.array([[0, 0], [100, 0]], dtype=np.float32))
    eng = Engine(m, lif, n_instances=1)
    st = State.initial(n, lif, 1)

    arrival = None
    for t in range(lif.delay_steps + 10):
        inj = np.zeros((n, 1), dtype=np.float32)
        if t == 0:
            inj[0, 0] = 100.0  # force neuron 0 to spike at t=0
        eng.step(st, inj)
        if st.g[1, 0] > 0 and arrival is None:
            arrival = t

    assert arrival == lif.delay_steps, (
        f"current arrived at step {arrival}, expected {lif.delay_steps} "
        f"({lif.delay_ms}ms at dt={lif.dt_ms}ms)"
    )


def test_delay_buffer_does_not_leak_between_instances():
    """The ring buffer is shared storage; a slot must stay per-instance."""
    lif = LIF()
    n = 2
    m = sp.csr_matrix(np.array([[0, 0], [100, 0]], dtype=np.float32))
    eng = Engine(m, lif, n_instances=6)
    st = State.initial(n, lif, 6)

    for t in range(lif.delay_steps + 5):
        inj = np.zeros((n, 6), dtype=np.float32)
        if t == 0:
            inj[0, 2] = 100.0  # only instance 2 spikes
        eng.step(st, inj)

    assert st.g[1, 2] > 0, "instance 2 should have received current"
    for i in (0, 1, 3, 4, 5):
        assert st.g[1, i] == 0, f"instance {i} received current it never sent"
