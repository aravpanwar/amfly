"""The closed loop: reward, punishment, habituation, escalating neglect.

This module is the heart of the redesign and had no tests at all. The
properties below are the ones the piece actually claims, so if any of them
breaks the claim in the README becomes false.

Fast: no connectome needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from amfly.config import CHAMBERS, OPERATOR
from amfly.wiring.compulsion import (
    NEGLECT_BELOW,
    REWARD_AT,
    Compulsion,
    resolve_reward_neurons,
)

N = 400
REWARD_IDX = np.arange(10, 40)
THERMO_IDX = np.arange(100, 125)


def _comp(**kw):
    return Compulsion(REWARD_IDX, THERMO_IDX, N, 6, **kw)


# --------------------------------------------------------------------------
# Isolation. The chambers must never be touched by this, whatever it does.
# --------------------------------------------------------------------------

def test_only_the_operator_column_is_ever_written():
    """The five are heated by the buttons. Nothing here may reach them."""
    c = _comp()
    for levels in ([1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [1, 0, 1, 0, 0.5]):
        inj = c.update(np.array(levels, dtype=np.float32))
        for i in CHAMBERS:
            assert np.all(inj[:, i] == 0), f"compulsion wrote chamber {i}"


def test_reward_and_punishment_reach_different_populations():
    """Reward goes to the dopaminergic cells, punishment to thermoreceptors."""
    c = _comp()
    for _ in range(50):
        inj = c.update(np.array([1, 1, 1, 1, 1], dtype=np.float32))
    assert np.all(inj[REWARD_IDX, OPERATOR] > 0), "reward never reached PAM"

    c2 = _comp()
    for _ in range(50):
        inj2 = c2.update(np.array([0, 0, 0, 0, 0], dtype=np.float32))
    assert np.all(inj2[THERMO_IDX, OPERATOR] > 0), "punishment never reached TRN_VP"


def test_untouched_neurons_stay_at_zero():
    c = _comp()
    for _ in range(30):
        inj = c.update(np.array([1, 0, 1, 0, 0], dtype=np.float32))
    other = np.setdiff1d(np.arange(N), np.union1d(REWARD_IDX, THERMO_IDX))
    assert np.all(inj[other, :] == 0)


# --------------------------------------------------------------------------
# Reward and punishment respond to the right thing.
# --------------------------------------------------------------------------

def test_maxed_chambers_pay_and_empty_ones_hurt():
    hot, cold = _comp(), _comp()
    for _ in range(40):
        hot.update(np.full(5, 1.0, dtype=np.float32))
        cold.update(np.zeros(5, dtype=np.float32))
    assert hot.state[0] > cold.state[0], "maxed chambers did not pay more"
    assert cold.state[1] > hot.state[1], "empty chambers did not hurt more"


def test_thresholds_are_respected():
    """Just under the line must not pay; just under neglect must not hurt."""
    c = _comp()
    for _ in range(30):
        c.update(np.full(5, REWARD_AT - 0.02, dtype=np.float32))
    assert c.state[0] == pytest.approx(0.0, abs=1e-3), "paid below REWARD_AT"

    c2 = _comp()
    for _ in range(30):
        c2.update(np.full(5, NEGLECT_BELOW + 0.02, dtype=np.float32))
    assert c2.state[1] == pytest.approx(0.0, abs=1e-3), "hurt above NEGLECT_BELOW"


# --------------------------------------------------------------------------
# Habituation. Holding the same chamber must stop paying; this is what broke
# the operator's habit of camping on two chambers forever.
# --------------------------------------------------------------------------

def test_holding_a_chamber_stops_paying():
    c = _comp()
    held = np.array([1, 0.6, 0.6, 0.6, 0.6], dtype=np.float32)
    early = [c.update(held) is not None or c.state[0] for _ in range(60)][-1]
    for _ in range(1500):
        c.update(held)
    late = c.state[0]
    assert late < early, (
        f"reward did not decay while holding: {early:.3f} -> {late:.3f}. "
        "Without habituation the operator camps on two chambers."
    )


def test_a_rested_chamber_recovers_its_value():
    c = _comp()
    held = np.array([1, 0.6, 0.6, 0.6, 0.6], dtype=np.float32)
    for _ in range(1500):
        c.update(held)
    saturated = float(c._habit[0])

    rested = np.array([0.6, 0.6, 0.6, 0.6, 0.6], dtype=np.float32)
    for _ in range(1500):
        c.update(rested)
    assert c._habit[0] < saturated, "habituation never recovered"


# --------------------------------------------------------------------------
# Escalating neglect. The bug this replaced: debt capped at 1.0 after 83ms,
# so every abandoned chamber carried identical debt and there was no gradient
# towards the worst one. Chamber 2 was never rescued once in three seconds.
# --------------------------------------------------------------------------

def test_neglect_debt_keeps_growing():
    c = _comp()
    lv = np.array([1, 1, 0, 1, 1], dtype=np.float32)
    for _ in range(900):
        c.update(lv)
    early = float(c._debt[2])
    for _ in range(9000):
        c.update(lv)
    late = float(c._debt[2])
    assert late > early * 1.5, (
        f"debt stopped growing: {early:.4f} -> {late:.4f}. It capped at 1.0 "
        "after 83ms before, which removed the gradient towards the worst "
        "chamber entirely."
    )


def test_the_longest_ignored_chamber_hurts_most():
    """There must be a gradient, or the operator cannot know which to rescue."""
    c = _comp()
    lv = np.array([1, 1, 0, 1, 1], dtype=np.float32)
    for _ in range(4000):
        c.update(lv)
    # Now neglect a second one as well, briefly.
    lv2 = np.array([1, 0, 0, 1, 1], dtype=np.float32)
    for _ in range(200):
        c.update(lv2)
    assert c._debt[2] > c._debt[1], (
        "the long-ignored chamber does not carry more debt than the "
        "recently-ignored one"
    )


def test_attention_pays_the_debt_down():
    c = _comp()
    for _ in range(4000):
        c.update(np.array([1, 1, 0, 1, 1], dtype=np.float32))
    owed = float(c._debt[2])
    for _ in range(400):
        c.update(np.array([1, 1, 1, 1, 1], dtype=np.float32))
    assert c._debt[2] < owed, "rescuing a chamber did not reduce its debt"


# --------------------------------------------------------------------------
# Determinism and wiring.
# --------------------------------------------------------------------------

def test_compulsion_is_deterministic():
    def run():
        c = _comp()
        out = []
        for t in range(500):
            lv = np.array([1, 0.3, 0.9, 0.2, 0.7], dtype=np.float32)
            c.update(lv)
            out.append(c.state)
        return out

    assert run() == run()


def test_empty_reward_population_raises():
    """A silent miss here would leave the loop half-open and still 'work'."""
    with pytest.raises(ValueError, match="dopaminergic"):
        Compulsion(np.array([], dtype=int), THERMO_IDX, N, 6)


def test_resolve_reward_neurons_matches_prefixes_only():
    types = np.array(["PAM01", "PPL101", "PPL201", "PAMx", "MBON01", "KC", "PA"])
    idx = resolve_reward_neurons(types)
    assert set(idx.tolist()) == {0, 1, 2, 3}
    assert 4 not in idx and 5 not in idx, "matched a non-dopaminergic type"
