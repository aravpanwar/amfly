"""Replay a finished run's operator spikes through Dials with new parameters.

The operator never reads the chambers, so its spike train is fixed. That means
button and compulsion parameters can be tested in seconds instead of the 12
minutes a fresh 3s run costs.

    python replay.py --run runs/paced --decide-every 1 --press 1800

Caveat worth stating: the compulsion loop DOES feed back into the operator in a
real run, so replayed heat is not what a fresh run would produce. It is a
pacing instrument, not a simulator. Anything chosen here gets confirmed by a
real run before it ships.
"""
import argparse, sys, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from amfly.config import LIF, CHAMBERS
from amfly.wiring.dials import Dials
from amfly.wiring.compulsion import Compulsion


def build_operator_matrix(run: Path):
    """(T, N_sub) bool of operator spikes, plus the neuron ids as columns."""
    d = np.load(run / "run.npz")
    m = d["events_instance"] == 5
    step = d["events_step"][m]
    neuron = d["events_neuron"][m]
    T = len(d["rates"])
    ids = np.unique(neuron)
    col = np.searchsorted(ids, neuron)
    mat = np.zeros((T, len(ids)), dtype=bool)
    mat[step, col] = True
    return mat, ids, T


def replay(run: Path, dn_indices, dn_weights, **params):
    mat, ids, T = build_operator_matrix(run)
    lif = LIF()
    dial = Dials.build(dn_indices, lif.dt_ms, window_ms=50.0,
                       dn_weights=dn_weights)
    dial.grip = params.pop("grip", 2)
    for k, v in params.items():
        setattr(dial, k, v)

    # Map the dial's global block indices into columns of `mat`. Neurons that
    # never spiked in this run are absent from `ids`; they contribute zero
    # either way, so dropping them is exact, not an approximation.
    blocks = []
    for b in dial._blocks:
        keep = np.isin(b, ids)
        blocks.append(np.searchsorted(ids, b[keep]))

    counts = np.zeros((T, len(CHAMBERS)), dtype=np.int32)
    for bi, pos in enumerate(blocks):
        if len(pos):
            counts[:, bi] = mat[:, pos].sum(axis=1)

    # Real indices so Compulsion's guard stays intact. Only _debt and
    # _habit are read here; the injection buffer is never used in replay.
    comp = Compulsion(np.arange(4, dtype=np.int64),
                      np.arange(4, dtype=np.int64), 8, 6)
    levels = np.zeros((T, len(CHAMBERS)), dtype=np.float32)

    # Drive Dials.update directly off the precomputed per-block counts by
    # writing them into its ring buffer, which is what update() would compute.
    for t in range(T):
        dial.set_debt(comp._debt)
        slot = t % dial.window_steps
        dial._counts[slot, :] = counts[t]
        _dials_step(dial, t)
        levels[t] = dial._levels
        comp.update(levels[t])
    return levels


def _dials_step(dial, step):
    """The body of Dials.update() after the spike counting, verbatim."""
    totals = dial._counts.sum(axis=0).astype(np.float64)
    dial._baseline *= dial._baseline_decay
    dial._baseline += (1.0 - dial._baseline_decay) * totals.mean()
    reference = max(float(dial._baseline.mean()), dial._baseline_floor)
    ratio = totals / reference
    target = np.clip((ratio - ratio.mean()) * dial.contrast, -1.0, 1.0)
    dial._queue.append((step + dial.latency_steps, target))
    applied = None
    while dial._queue and dial._queue[0][0] <= step:
        _, applied = dial._queue.pop(0)
    if applied is not None and dial.grip and dial.grip < dial.n_blocks:
        sc = applied + dial.debt_bias * dial._debt_view
        held = set(int(i) for i in dial._held)
        challengers = [i for i in range(dial.n_blocks) if i not in held]
        if challengers:
            weakest = min(held, key=lambda i: sc[i])
            best = max(challengers, key=lambda i: sc[i])
            if sc[best] > sc[weakest] + dial.switch_margin:
                held.discard(weakest)
                held.add(best)
                dial._held = np.array(sorted(held))
    if applied is not None and dial.buttons:
        if dial.grip and dial.grip < dial.n_blocks:
            pressed = np.zeros(dial.n_blocks, dtype=bool)
            pressed[dial._held] = True
        else:
            pressed = applied > np.median(applied)
        delta = np.where(pressed, dial.press_rate, -dial.release_rate)
        dial._levels += delta.astype(np.float32)
        np.clip(dial._levels, 0.15, 1.0, out=dial._levels)


def score(levels, frame_ms=33.0):
    """Measure what a VIEWER sees, at a frame, not at a 0.1ms step.

    The per-step version of `motion` was measuring the press rate, not the
    system: any press slower than 1/1800 moves less than the 0.002 threshold in
    one step, so motion read exactly 0.00 for every slow setting while the
    chambers were in fact moving perfectly well. A metric that returns a hard
    zero across wildly different settings is wrong; that lesson cost an evening
    once already.

    Sampling at a frame interval asks the question that matters: between two
    frames of the clip, did a line visibly move?
    """
    lv = levels / max(float(levels.max()), 1e-9)
    stride = max(1, int(round(frame_ms / 0.1)))
    fr = lv[::stride]
    # A "stroke" is a contiguous run of one chamber moving the same direction,
    # measured per frame so a stroke is something a viewer could follow.
    sign = np.sign(np.diff(fr, axis=0))
    switches = int((np.diff(sign, axis=0) != 0).sum())
    T = len(lv)
    secs = T * 0.1 / 1000.0
    # Visible motion: a line moved at least half a percent of full scale
    # between two frames, which is about 1px on a 200px-tall plot.
    return {
        "spread": round(float(np.mean(lv.max(axis=1) - lv.min(axis=1))), 3),
        "motion": round(float((np.abs(np.diff(fr, axis=0)).max(axis=1) > 0.005).mean()), 3),
        "pinned": round(float(((lv > 0.97) | (lv < 0.18)).mean()), 3),
        "switch_per_s": round(switches / secs, 1),
        "travel": round(float(np.abs(np.diff(fr, axis=0)).sum() / secs), 2),
        "means": [int(v * 100) for v in lv.mean(axis=0)],
        "peaks": [int(v * 100) for v in lv.max(axis=0)],
        "above_half": [int(v * 100) for v in (lv > 0.5).mean(axis=0)],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=ROOT / "runs/paced")
    ap.add_argument("--dn", type=Path, default=ROOT / "runs/dn.npz",
                    help="cached DN indices and synaptic weights, so the "
                         "1.1GB weights table is not reloaded per experiment")
    ap.add_argument("--build-dn", type=Path, default=None,
                    help="path to the connectome; builds the DN cache and exits")
    ap.add_argument("--margin", type=float, default=0.40)
    ap.add_argument("--press", type=float, default=2000.0)
    args = ap.parse_args()

    if args.build_dn is not None:
        from amfly.data.loader import load
        c = load(args.build_dn)
        dn = c.descending_indices()
        w = np.abs(c.csr[:, dn]).sum(axis=0).A.ravel()
        args.dn.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.dn, dn=dn, w=w)
        print("cached %d DNs to %s" % (len(dn), args.dn))
        return 0

    d = np.load(args.dn)
    lv = replay(args.run, d["dn"], d["w"], switch_margin=args.margin,
                press_rate=1.0 / args.press,
                release_rate=1.0 / (args.press * 2.5))
    for k, v in score(lv).items():
        print("%-14s %s" % (k, v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
