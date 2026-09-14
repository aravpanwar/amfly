"""The divergence metric itself.

Fast: no connectome needed. These guard the property that made the first run
unreadable, namely that equal spike counts can hide completely different spike
identities.
"""

from __future__ import annotations

import numpy as np

from amfly.sim.divergence import Divergence, spike_identity_distance


def test_identical_spikes_are_zero_distance():
    a = np.zeros(100, dtype=bool)
    a[[1, 5, 9]] = True
    assert spike_identity_distance(a, a.copy()) == 0


def test_equal_counts_different_neurons_is_not_zero():
    """The exact failure that made the first run unreadable.

    Same number of spikes, entirely different neurons. A rate metric calls
    these identical; they are not.
    """
    a = np.zeros(100, dtype=bool)
    b = np.zeros(100, dtype=bool)
    a[[1, 2, 3]] = True
    b[[7, 8, 9]] = True
    assert a.sum() == b.sum()
    assert spike_identity_distance(a, b) == 6


def test_reference_instance_is_always_zero():
    spikes = np.zeros((50, 6), dtype=bool)
    spikes[::3] = True
    d = Divergence(reference=2)
    out = d.update(spikes, 0)
    assert out[2] == 0


def test_first_divergence_records_earliest_step_only():
    d = Divergence(reference=0)

    same = np.zeros((20, 6), dtype=bool)
    same[[1, 2]] = True
    d.update(same, 0)
    assert d.first_divergence == {}

    diff = same.copy()
    diff[5, 3] = True
    d.update(diff, 1)
    assert d.first_divergence == {3: 1}

    diff2 = same.copy()
    diff2[6, 3] = True
    d.update(diff2, 2)
    assert d.first_divergence == {3: 1}, "onset must not be overwritten"


def test_cumulative_is_monotonic():
    d = Divergence(reference=0)
    base = np.zeros((40, 6), dtype=bool)
    base[[1, 2, 3]] = True
    for t in range(10):
        s = base.copy()
        s[t, 4] = True
        d.update(s, t)
    cum = d.cumulative()
    assert np.all(np.diff(cum[:, 4]) >= 0)
    assert cum[-1, 4] > 0
    assert np.all(cum[:, 0] == 0)


def test_unheated_instances_stay_at_zero_together():
    """Isolation, at the metric level: only the perturbed column moves."""
    d = Divergence(reference=0)
    s = np.zeros((80, 6), dtype=bool)
    s[[4, 8, 12]] = True
    s[20, 3] = True  # only instance 3 perturbed
    out = d.update(s, 0)
    assert out[0] == 0
    assert out[1] == 0 and out[2] == 0 and out[4] == 0 and out[5] == 0
    assert out[3] == 1
