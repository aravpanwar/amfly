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
from amfly.data.loader import load, soma_positions, verify_counts  # noqa: E402
from amfly.io.spikes import Recorder  # noqa: E402
from amfly.sim.backend import configure_torch_determinism, describe  # noqa: E402
from amfly.sim.divergence import Divergence  # noqa: E402
from amfly.sim.engine import Engine, State  # noqa: E402
from amfly.wiring.chambers import ContinuousHeat, Electrode  # noqa: E402
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
    ap.add_argument("--debt-bias", type=float, default=None,
                    help="how strongly neglect pulls the operator towards a "
                         "chamber it has been ignoring")
    ap.add_argument("--contrast", type=float, default=None,
                    help="scales the differences between DN blocks")
    ap.add_argument("--escalate", type=float, default=None,
                    help="how fast neglect debt accumulates")
    ap.add_argument("--switch-margin", type=float, default=None,
                    help="how decisively a chamber must win before it takes a "
                         "button from one already held. Low values let the "
                         "middle chambers braid together at mid-range")
    ap.add_argument("--press-rate", type=float, default=None,
                    help="steps for a button to raise a chamber through its "
                         "full range, e.g. 2000 for 200ms at dt=0.1ms")
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
    ap.add_argument("--heat", action="store_true",
                    help="use the original thermal channel into 25 TRN_VP "
                         "thermoreceptors instead of the electrode")
    ap.add_argument("--no-convulse", action="store_true",
                    help="electrode without the direct motor drive, so the "
                         "bodies stay still")
    ap.add_argument("--motor-mv", type=float, default=60.0,
                    help="amplitude into the 708 VNC motor neurons. Measured "
                         "against the real baseline: 1.20x at 10, 1.43x at "
                         "25, 1.75x at 60")
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
    if args.debt_bias is not None:
        dial.debt_bias = args.debt_bias
    if args.contrast is not None:
        dial.contrast = args.contrast
    if args.switch_margin is not None:
        dial.switch_margin = args.switch_margin
    if args.press_rate is not None:
        # Keep release slower than press, in the measured 2.5x ratio, so a
        # chamber still falls more slowly than it rises.
        dial.press_rate = 1.0 / args.press_rate
        dial.release_rate = 1.0 / (args.press_rate * 2.5)
    _loads = [float(dn_out[np.searchsorted(dn, b)].sum()) for b in dial._blocks]
    log.info("DN block synaptic load: %s (spread %.2fx)",
             [int(v) for v in _loads], max(_loads) / max(min(_loads), 1))
    if args.dial_latency_ms is not None:
        dial.latency_steps = max(1, int(round(args.dial_latency_ms / lif.dt_ms)))
        log.info("dial latency overridden to %.0f ms", args.dial_latency_ms)
    # Electrical stimulation, not heat.
    #
    # Heat implied cooking, which this model cannot represent: no tissue
    # damage, no nociception, nothing that degrades. What the simulation
    # literally does is inject millivolts of depolarising current, so an
    # electrode is the accurate description and millivolts the real unit.
    #
    # The electrode has a POSITION and stimulates by distance, which is what
    # makes it defensible: 5,579 neurons across seven superclasses rather than
    # 25 cells of one type, graded by a Gaussian falloff. A shock does not
    # select for function. The thermoreceptors it replaces have no soma
    # coordinates at all, 0 of 25, so the old channel could never have been
    # placed in space even in principle.
    if args.heat:
        heat = ContinuousHeat(th, c.n, 6)
        log.info("stimulus: legacy thermal channel, %d TRN_VP neurons", len(th))
    else:
        xyz = soma_positions(c.body_ids)
        types = c.cell_type.astype(str)
        esc = np.flatnonzero(np.char.startswith(types, "DNp01"))
        if len(esc) == 0:
            raise SystemExit("DNp01 not found; cannot site the electrode")
        site = xyz[esc[0]]
        motor = (
            None if args.no_convulse
            else np.flatnonzero(c.superclass.astype(str) == "vnc_motor")
        )
        heat = Electrode(
            soma_xyz=xyz, site=site, n_neurons=c.n, n_instances=6,
            motor_indices=motor, motor_mv=args.motor_mv,
        )
        log.info(
            "stimulus: electrode at %s, %d neurons reached, motor drive %s",
            site.round(0).tolist(), heat.reached,
            "off" if motor is None else f"{len(motor)} at {args.motor_mv:.0f}mV",
        )
    rec = Recorder.build(c.n, 6, dn, th, sample=args.record_sample)

    # Close the loop: chamber state reaches the operator. Reward through the
    # real dopaminergic populations, punishment through its own
    # thermoreceptors. See amfly/wiring/compulsion.py.
    comp = None
    if not args.no_compulsion:
        rew = resolve_reward_neurons(c.cell_type)
        comp = Compulsion(rew, th, c.n, 6)
        if args.escalate is not None:
            comp.escalate_rate = args.escalate
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
        heat_inj = (heat.injection(dial.levels) if args.heat
                    else heat.injection(dial.levels, t))
        inj = zeros()
        if use_cuda:
            inj += torch.from_numpy(heat_inj).to("cuda")
        else:
            inj += heat_inj
        if t % period < width:
            inj[drive, :] += np.float32(args.baseline_mv)

        if comp is not None:
            # Close the loop: what the operator is burned for also biases
            # which buttons it can press.
            dial.set_debt(comp._debt)
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
        rw, pn = comp.state if comp is not None else (None, None)
        rec.record(t, spikes, heat.levels, pos, ham, reward=rw, punish=pn,
                   held=dial.held)

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
                "switch_margin": dial.switch_margin,
                "press_rate_steps": round(1.0 / dial.press_rate),
                "compulsion": comp.summary() if comp is not None else None,
            },
            indent=2,
        )
    )
    log.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
