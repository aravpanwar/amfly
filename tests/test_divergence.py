"""Phase 2 gate, on the real connectome.

project.md calls this "the whole piece": the five must diverge only from
differing heat. Everything before it is infrastructure and everything after it
is rendering, so this is the test that decides whether the project is real.

Slow (loads 1.1GB, runs the full 166,700-neuron network) and skipped when the
data is absent.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from amfly.config import LIF, OPERATOR
from amfly.data.loader import load
from amfly.sim.engine import Engine, State
from amfly.wiring.chambers import Heat

DATA_DIR = Path(os.environ.get("AMFLY_DATA", "data"))

pytestmark = [
    pytest.mark.skipif(
        not (DATA_DIR / "weights.feather").exists(),
        reason=f"connectome not found in {DATA_DIR}; set AMFLY_DATA",
    ),
    pytest.mark.slow,
]

STEPS = 250
HEATED = 0


@pytest.fixture(scope="module")
def connectome():
    return load(DATA_DIR)


def _run(c, heat_on: bool) -> np.ndarray:
    lif = LIF()
    eng = Engine(c.csr, lif)
    st = State.initial(c.n, lif)
    th = c.thermo_indices()
    heat = Heat(th, c.n, 6)
    rates = np.zeros((STEPS, 6), dtype=np.int64)
    for t in range(STEPS):
        inj = (
            heat.injection(HEATED)
            if heat_on
            else np.zeros((c.n, 6), dtype=np.float32)
        )
        inj[th, :] += 6.0  # identical baseline to all six
        rates[t] = eng.step(st, inj).sum(axis=0)
    return rates


@pytest.fixture(scope="module")
def runs(connectome):
    return _run(connectome, True), _run(connectome, False)


def test_without_heat_all_six_are_identical(runs):
    """t=0 onward, nothing separates them. Measured: 397,994 spikes each."""
    _, control = runs
    for i in range(1, 6):
        assert np.array_equal(control[:, 0], control[:, i])
    assert control.sum() > 0, "network silent; test passed vacuously"


def test_heat_isolates_to_one_chamber(runs):
    """The fly_brain_food anti-pattern would fail here."""
    hot, control = runs
    for i in (1, 2, 3, 4, OPERATOR):
        assert np.array_equal(hot[:, i], control[:, i]), (
            f"instance {i} changed when only chamber {HEATED} was heated"
        )


def test_heated_chamber_actually_diverges(runs):
    """Divergence is caused, not drifted. Measured onset: step 216."""
    hot, control = runs
    assert not np.array_equal(hot[:, HEATED], control[:, HEATED])
    onset = np.flatnonzero(hot[:, HEATED] != control[:, HEATED])
    assert len(onset) > 0
    assert onset[0] < STEPS, "divergence must occur within the window"


def test_operator_is_never_heated(connectome):
    """It sits outside. The dial is the only thing connecting it to the five."""
    heat = Heat(connectome.thermo_indices(), connectome.n, 6)
    for pos in range(5):
        inj = heat.injection(pos)
        assert np.all(inj[:, OPERATOR] == 0.0)
        assert heat.levels[OPERATOR] == 0.0
