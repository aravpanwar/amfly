"""Run the piece and dump the result.

    python scripts/run_sim.py --ms 500 --out runs/001

Six instances of one connectome. Five in chambers, one at a dial. The operator's
descending activity selects which chamber is heated. It receives no feedback.

Slow on CPU by design: roughly 150ms per 0.1ms step at 166,700 neurons, so a
second of simulated time is about 25 minutes. The deterministic pull model is
the reason, and it is a deliberate trade. See amfly/sim/engine.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from amfly.config import CHAMBERS, LIF, OPERATOR  # noqa: E402
from amfly.data.loader import load, verify_counts  # noqa: E402
from amfly.io.spikes import Recorder  # noqa: E402
from amfly.sim.backend import configure_torch_determinism, describe  # noqa: E402
from amfly.sim.divergence import Divergence  # noqa: E402
from amfly.sim.engine import Engine, State  # noqa: E402
from amfly.wiring.chambers import ContinuousHeat  # noqa: E402
from amfly.wiring.compulsion import (  # noqa: E402
    Compulsion, resolve_reward_neurons,
)
from amfly.wiring.dials import Dials  # noqa: E402

log = logging.getLogger("amfly")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("runs/latest"))
    ap.add_argument("--ms", type=float, default=100.0, help="simulated milliseconds")
    ap.add_argument("--baseline-mv", type=float, default=5.0,
                    help="phasic drive amplitude, identical to all six")
    ap.add_argument("--grip", type=int, default=2,
                    help="dials the operator can hold at once; 5 removes the "
                         "limit and restores the original indifferent design")
    ap.add_argument("--no-compulsion", action="store_true",
                    help="original design: the operator gets no feedback at all")
    ap.add_argument("--record-sample", type=int, default=2000,
                    help="neurons recorded at full resolution, for the brain "
                         "view. 2000 lights only ~500 of 24000 rendered points")
    ap.add_argument("--dial-window-ms", type=float, default=50.0)
    ap.add_argument("--dial-latency-ms", type=float, default=None,
                    help="override the 500ms authored latency. Shorten it to "
                         "see switches in a short run; the 500ms default is "
                         "the authored value for a finished clip.")
    ap.add_argument("--pulse-period-ms", type=float, default=10.0,
                    help="phasic drive period; tonic drive synchronises the "
                         "network and suppresses divergence")
    ap.add_argument("--pulse-width-ms", type=float, default=0.5)
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--cpu", action="store_true",
                    help="force the numpy reference backend even if CUDA works")
    ap.add_argument("--silence-unclear-nt", action="store_true",
                    help="sensitivity run: silence the 3,177 neurons (1.9%%) "
                         "whose neurotransmitter is unclear or missing, rather "
                         "than defaulting them to excitatory")
    ap.add_argument("--weight-threshold", type=int, default=1,
                    help="minimum synapse count per connection. 1 keeps all "
                         "25,582,938 edges; anything higher must be declared")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    lif = LIF()
    steps = int(round(args.ms / lif.dt_ms))

    c = load(
        args.data,
        weight_threshold=args.weight_threshold,
        silence_unclear_nt=args.silence_unclear_nt,
    )
    if args.no_verify:
        pass
    elif args.weight_threshold > 1:
        log.warning(
            "weight threshold %d is set, so the published counts do not apply "
            "and the count gate is skipped. Declare this in any result.",
            args.weight_threshold,
        )
    else:
        verify_counts(c)
        log.info("counts verified")

    dn = c.descending_indices()
    th = c.thermo_indices()
    log.info("descending neurons: %d, thermoreceptors: %d", len(dn), len(th))

    use_cuda = (not args.cpu) and configure_torch_determinism()
    if use_cuda:
        from amfly.sim.engine_cuda import CudaEngine
        import torch

        eng = CudaEngine(c.csr, lif)
        st = eng.initial_state()
        zeros = lambda: torch.zeros((c.n, 6), dtype=torch.float32, device="cuda")
        to_np = lambda x: x.cpu().numpy()
    else:
        eng = Engine(c.csr, lif)
        st = State.initial(c.n, lif)
        zeros = lambda: np.zeros((c.n, 6), dtype=np.float32)
        to_np = lambda x: x
    # Balance the DN blocks by synaptic output. Contiguous bodyId slices give
    # a 6.70x spread, so two chambers dominate the dial regardless of what the
    # operator does. See amfly/wiring/dials.py.
    dn_out = np.abs(c.csr[:, dn]).sum(axis=0).A.ravel()
    dial = Dials.build(dn, lif.dt_ms, window_ms=args.dial_window_ms,
                       dn_weights=dn_out)
    dial.grip = None if args.grip >= 5 else args.grip
    _loads = [float(dn_out[np.searchsorted(dn, b)].sum()) for b in dial._blocks]
    log.info("DN block synaptic load: %s (spread %.2fx)",
             [int(v) for v in _loads], max(_loads) / max(min(_loads), 1))
    if args.dial_latency_ms is not None:
        dial.latency_steps = max(1, int(round(args.dial_latency_ms / lif.dt_ms)))
        log.info("dial latency overridden to %.0f ms", args.dial_latency_ms)
    heat = ContinuousHeat(th, c.n, 6)
    rec = Recorder.build(c.n, 6, dn, th, sample=args.record_sample)

    # Close the loop: chamber state reaches the operator. Reward through the
    # real dopaminergic populations, punishment through its own
    # thermoreceptors. See amfly/wiring/compulsion.py.
    comp = None
    if not args.no_compulsion:
        rew = resolve_reward_neurons(c.cell_type)
        comp = Compulsion(rew, th, c.n, 6)
        log.info("compulsion: %d reward neurons (PAM/PPL1/PPL2), grip %s",
                 len(rew), dial.grip or "unlimited")
    div = Divergence()

    # Phasic drive into a strided slice of the central-brain sensory neurons.
    # Driving the 25 thermoreceptors tonically made the whole network ring at
    # the drive rate, and a globally synchronised network swamps perturbations.
    # See docs/negative-results.md.
    sensory = np.flatnonzero(
        np.char.startswith(c.superclass.astype(str), "cb_sensory")
    )
    drive = sensory[::7]
    period = max(1, int(round(args.pulse_period_ms / lif.dt_ms)))
    width = max(1, int(round(args.pulse_width_ms / lif.dt_ms)))
    log.info("phasic drive: %d neurons, %d-step pulse every %d steps",
             len(drive), width, period)

    log.info("running %d steps (%.1f ms simulated)", steps, args.ms)
    t0 = time.time()
    for t in range(steps):
        heat_inj = heat.injection(dial.levels)
        inj = zeros()
        if use_cuda:
            inj += torch.from_numpy(heat_inj).to("cuda")
        else:
            inj += heat_inj
        if t % period < width:
            inj[drive, :] += np.float32(args.baseline_mv)

        if comp is not None:
            comp_inj = comp.update(heat.levels / 8.0)
            if use_cuda:
                inj += torch.from_numpy(comp_inj).to("cuda")
            else:
                inj += comp_inj

        spikes_dev = eng.step(st, inj)
        spikes = to_np(spikes_dev)
        levels = dial.update(spikes, t)
        pos = int(np.argmax(levels))  # hottest chamber, for the summary only
        ham = div.update(spikes, t)
        rec.record(t, spikes, heat.levels, pos, ham)

        if t % 100 == 0 and t:
            el = time.time() - t0
            log.info(
                "  step %d/%d  %.0f ms/step  heat=%s  rates=%s",
                t, steps, el / t * 1000,
                (levels * 100).astype(int), spikes.sum(axis=0),
            )

    elapsed = time.time() - t0
    log.info("done in %.1fs (%.0f ms/step)", elapsed, elapsed / steps * 1000)

    lv = dial.level_history()
    log.info("final heat levels: %s", (lv[-1] * 100).round(0))
    log.info("max level reached per chamber: %s", (lv.max(axis=0) * 100).round(0))
    if div.first_divergence:
        for i in sorted(div.first_divergence):
            log.info("  instance %d diverged at step %d",
                     i, div.first_divergence[i])
    else:
        log.info("  no instance diverged; this is a negative result, record it")
    log.info("final cumulative distance: %s", div.cumulative()[-1])
    identical = not div.first_divergence
    if lv.max() == 0.0:
        log.warning(
            "no chamber was ever heated: latency is %.0f ms, so run at least "
            "that long",
            lif.delay_ms + dial.latency_steps * lif.dt_ms,
        )

    path = rec.save(args.out)
    (args.out / "provenance.json").write_text(
        json.dumps(
            {
                **c.provenance,
                "steps": steps,
                "ms": args.ms,
                "first_divergence": div.first_divergence,
                "final_levels": dial.levels.tolist(),
                "max_levels": dial.level_history().max(axis=0).tolist(),
                "seconds": round(elapsed, 1),
                "backend": describe(),
                "grip": dial.grip,
                "compulsion": comp.summary() if comp is not None else None,
            },
            indent=2,
        )
    )
    log.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
