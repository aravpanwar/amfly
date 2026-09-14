"""Run classification.

These guard against the two ways a run could be misreported: calling a
momentary coincidence a re-convergence, and failing to notice that instances
drifted for reasons other than heat.
"""

from __future__ import annotations

import numpy as np

from amfly.sim.analysis import analyse, unheated_stayed_identical


def _ham(T=1000, n=6):
    return np.zeros((T, n), dtype=np.int32)


def test_never_diverged_is_reported_as_such():
    v = analyse(_ham(), instance=3)
    assert not v.diverged
    assert v.onset_step is None
    assert "never diverged" in v.summary()


def test_sustained_divergence():
    h = _ham()
    h[100:, 3] = 250
    v = analyse(h, instance=3)
    assert v.diverged
    assert v.onset_step == 100
    assert not v.reconverged
    assert v.final_distance == 250
    assert v.peak_distance == 250
    assert "stayed separated" in v.summary()


def test_reconvergence_is_detected_and_named_a_negative_result():
    h = _ham()
    h[100:300, 3] = 40
    # zero from 300 to 1000, a 700-step hold
    v = analyse(h, instance=3, hold_steps=200)
    assert v.diverged
    assert v.onset_step == 100
    assert v.reconverged
    assert v.reconvergence_step == 300
    assert "negative result" in v.summary()


def test_brief_coincidence_is_not_reconvergence():
    """Two trajectories can agree for a moment without having merged."""
    h = _ham()
    h[100:, 3] = 40
    h[500:520, 3] = 0  # 20 steps of coincidence, then diverged again
    v = analyse(h, instance=3, hold_steps=200)
    assert v.diverged
    assert not v.reconverged


def test_short_zero_tail_is_not_reconvergence():
    h = _ham()
    h[100:, 3] = 40
    h[-50:, 3] = 0  # only 50 steps of zeros at the end
    v = analyse(h, instance=3, hold_steps=200)
    assert not v.reconverged


def test_unheated_identical_passes_when_only_heated_moves():
    h = _ham()
    h[50:, 0] = 300  # chamber 0 heated, reference is chamber 1
    assert unheated_stayed_identical(h, heated=0, reference=1)


def test_unheated_identical_fails_on_stray_drift():
    """The failure that would invalidate every plot from a run."""
    h = _ham()
    h[50:, 0] = 300
    h[400, 4] = 1  # a single stray difference in an unheated chamber
    assert not unheated_stayed_identical(h, heated=0, reference=1)
