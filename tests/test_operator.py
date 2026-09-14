"""The dial, and the chamber heat it drives.

Fast: no connectome needed. The properties under test are the ones the piece
makes claims about, namely that the mapping is deterministic and auditable and
that the operator is never affected by what it does.
"""

from __future__ import annotations

import numpy as np
import pytest

from amfly.config import CHAMBERS, OPERATOR
from amfly.wiring.chambers import Heat
from amfly.wiring.operator import Dial

N = 500
DN = np.arange(0, 100)


def _spikes(active=None) -> np.ndarray:
    s = np.zeros((N, 6), dtype=bool)
    if active is not None:
        s[active, OPERATOR] = True
    return s


def test_dial_reads_only_the_operator_column():
    """Chamber activity must never move the dial."""
    d = Dial.build(DN, dt_ms=0.1, window_ms=5.0)
    s = np.zeros((N, 6), dtype=bool)
    s[80:100, 0] = True  # heavy activity in a CHAMBER, block 4
    for t in range(200):
        d.update(s, t)
    assert d.position == 0, "chamber activity moved the dial"


def test_dial_points_at_the_most_active_block():
    d = Dial.build(DN, dt_ms=0.1, window_ms=5.0)
    d.dwell_steps = 1
    # DN indices 60..79 are block 3 of 5 over 0..99.
    s = _spikes(np.arange(60, 80))
    for t in range(d.latency_steps + 10):
        d.update(s, t)
    assert d.position == 3


def test_dial_is_deterministic_across_runs():
    def run():
        d = Dial.build(DN, dt_ms=0.1, window_ms=5.0)
        d.dwell_steps = 1
        s = _spikes(np.arange(20, 40))
        out = [d.update(s, t) for t in range(d.latency_steps + 50)]
        return out, d.history

    a_pos, a_hist = run()
    b_pos, b_hist = run()
    assert a_pos == b_pos
    assert a_hist == b_hist


def test_ties_break_to_lowest_index():
    """No RNG, and no arbitrary choice: equal blocks resolve to the lowest."""
    d = Dial.build(DN, dt_ms=0.1, window_ms=5.0)
    d.dwell_steps = 1
    s = np.zeros((N, 6), dtype=bool)  # all blocks equally silent
    for t in range(d.latency_steps + 10):
        d.update(s, t)
    assert d.position == 0


def test_dial_latency_delays_the_change():
    """The change must not land before the latency has elapsed."""
    d = Dial.build(DN, dt_ms=0.1, window_ms=5.0)
    d.dwell_steps = 1
    s = _spikes(np.arange(80, 100))  # block 4
    for t in range(d.latency_steps - 2):
        assert d.update(s, t) == 0, "dial moved before the latency elapsed"
    for t in range(d.latency_steps - 2, d.latency_steps + 5):
        d.update(s, t)
    assert d.position == 4


def test_heat_reaches_only_the_selected_chamber():
    targets = np.arange(10, 20)
    h = Heat(targets, N, 6, ramp_steps=10)
    for _ in range(20):
        inj = h.injection(2)
    assert h.levels[2] > 0
    for c in CHAMBERS:
        if c != 2:
            assert h.levels[c] == 0.0
    assert np.all(inj[:, 0] == 0.0)
    assert np.all(inj[:, 2][targets] > 0.0)


def test_operator_is_never_heated():
    targets = np.arange(10, 20)
    h = Heat(targets, N, 6, ramp_steps=10)
    for pos in CHAMBERS:
        for _ in range(20):
            inj = h.injection(pos)
        assert h.levels[OPERATOR] == 0.0
        assert np.all(inj[:, OPERATOR] == 0.0)


def test_heat_only_touches_target_neurons():
    targets = np.arange(10, 20)
    h = Heat(targets, N, 6, ramp_steps=5)
    for _ in range(20):
        inj = h.injection(1)
    non_targets = np.setdiff1d(np.arange(N), targets)
    assert np.all(inj[non_targets, :] == 0.0)


def test_empty_thermo_targets_raise():
    """An empty target set would leave the heat channel silently dead.

    That is the most expensive failure available here: the five chambers would
    look deterministic for an uninteresting reason and the run would appear to
    succeed.
    """
    with pytest.raises(ValueError, match="thermosensory"):
        Heat(np.array([], dtype=int), N, 6)


def test_dial_reaches_a_quiet_block_when_it_becomes_unusually_active():
    """The fix for the two-of-five coverage failure.

    A block with a persistently lower rate must still be selectable when it
    rises relative to its own history. Under raw argmax it never could, and
    three of the five chambers were never heated.
    """
    d = Dial.build(DN, dt_ms=0.1, window_ms=5.0)
    d.latency_steps = 5
    d.dwell_steps = 10  # real default is 3000; shortened to fit the window

    loud = np.arange(0, 20)    # block 0
    quiet = np.arange(60, 80)  # block 3

    # Long stretch where block 0 dominates outright.
    s = _spikes(loud)
    for t in range(400):
        d.update(s, t)
    assert d.position == 0

    # Block 3 becomes active. It is no louder than block 0 ever was, but it is
    # far above its own baseline, so it must win.
    s2 = np.zeros((N, 6), dtype=bool)
    s2[loud, OPERATOR] = True
    s2[quiet, OPERATOR] = True
    reached = False
    for t in range(400, 700):
        if d.update(s2, t) == 3:
            reached = True
            break
    assert reached, "a quiet block never became selectable; coverage is broken"


def test_normalised_readout_is_still_deterministic():
    """The baseline is state, so confirm it does not introduce run-to-run drift."""
    def run():
        d = Dial.build(DN, dt_ms=0.1, window_ms=5.0)
        d.latency_steps = 5
        d.dwell_steps = 10
        out = []
        for t in range(300):
            s = _spikes(np.arange(20, 40) if t % 50 < 25 else np.arange(60, 80))
            out.append(d.update(s, t))
        return out

    assert run() == run()


def test_dial_holds_a_choice_for_the_dwell_period():
    """The fix for the chatter.

    Run r003 produced 725 switches with a median hold of 4 steps, because a
    fresh decision was computed every step and the latency only delayed the
    stream rather than pacing it. A committed choice must now stand.
    """
    d = Dial.build(DN, dt_ms=0.1, window_ms=5.0)
    d.latency_steps = 5
    d.dwell_steps = 200

    # Alternate the dominant block every single step, the worst case for chatter.
    a, b = np.arange(0, 20), np.arange(80, 100)
    positions = []
    for t in range(1200):
        positions.append(d.update(_spikes(a if t % 2 else b), t))

    changes = sum(
        1 for i in range(1, len(positions)) if positions[i] != positions[i - 1]
    )
    assert changes <= 1200 // d.dwell_steps + 2, (
        f"{changes} changes in 1200 steps with dwell={d.dwell_steps}; "
        "the dial is still chattering"
    )


def test_dwell_default_is_long_enough_for_heat_to_register():
    """Dwell must exceed the heat ramp, or a selection cannot leave a mark."""
    from amfly.wiring.chambers import Heat

    d = Dial.build(DN, dt_ms=0.1, window_ms=50.0)
    ramp = Heat(np.arange(5), 100, 6).ramp_steps
    assert d.dwell_steps > ramp, (
        f"dwell {d.dwell_steps} is shorter than the heat ramp {ramp}, so a "
        "chamber would be deselected before it is fully heated"
    )


def test_queued_switch_still_lands_during_dwell():
    """Dwell gates new decisions only; a queued one must not be stranded."""
    d = Dial.build(DN, dt_ms=0.1, window_ms=5.0)
    d.latency_steps = 50
    d.dwell_steps = 500

    s = _spikes(np.arange(80, 100))  # block 4
    final = 0
    for t in range(400):
        final = d.update(s, t)
    assert final == 4, "the queued switch never landed while dwell was active"
