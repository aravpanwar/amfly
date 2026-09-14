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
from amfly.wiring.chambers import Heat  # noqa: E402
from amfly.wiring.operator import Dial  # noqa: E402

log = logging.getLogger("amfly")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("runs/latest"))
    ap.add_argument("--ms", type=float, default=100.0, help="simulated milliseconds")
    ap.add_argument("--baseline-mv", type=float, default=5.0,
                    help="phasic drive amplitude, identical to all six")
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
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    lif = LIF()
    steps = int(round(args.ms / lif.dt_ms))

    c = load(args.data)
    if not args.no_verify:
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
    dial = Dial.build(dn, lif.dt_ms, window_ms=args.dial_window_ms)
    if args.dial_latency_ms is not None:
        dial.latency_steps = max(1, int(round(args.dial_latency_ms / lif.dt_ms)))
        log.info("dial latency overridden to %.0f ms", args.dial_latency_ms)
    heat = Heat(th, c.n, 6)
    rec = Recorder.build(c.n, 6, dn, th)
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
        heat_inj = heat.injection(dial.position)
        inj = zeros()
        if use_cuda:
            inj += torch.from_numpy(heat_inj).to("cuda")
        else:
            inj += heat_inj
        if t % period < width:
            inj[drive, :] += np.float32(args.baseline_mv)

        spikes_dev = eng.step(st, inj)
        spikes = to_np(spikes_dev)
        pos = dial.update(spikes, t)
        ham = div.update(spikes, t)
        rec.record(t, spikes, heat.levels, pos, ham)

        if t % 100 == 0 and t:
            el = time.time() - t0
            log.info(
                "  step %d/%d  %.0f ms/step  dial=%d  rates=%s",
                t, steps, el / t * 1000, pos, spikes.sum(axis=0),
            )

    elapsed = time.time() - t0
    log.info("done in %.1fs (%.0f ms/step)", elapsed, elapsed / steps * 1000)

    log.info("dial switches: %d", len(dial.history))
    if div.first_divergence:
        for i in sorted(div.first_divergence):
            log.info("  instance %d diverged at step %d",
                     i, div.first_divergence[i])
    else:
        log.info("  no instance diverged; this is a negative result, record it")
    log.info("final cumulative distance: %s", div.cumulative()[-1])
    identical = not div.first_divergence
    if identical and len(dial.history) == 0:
        log.warning(
            "no dial movement yet: latency is %.0f ms, so run at least that long",
            lif.delay_ms + dial.latency_steps * lif.dt_ms,
        )

    path = rec.save(args.out)
    (args.out / "provenance.json").write_text(
        json.dumps(
            {
                **c.provenance,
                "steps": steps,
                "ms": args.ms,
                "dial_switches": len(dial.history),
                "first_divergence": div.first_divergence,
                "dial_history": dial.history,
                "seconds": round(elapsed, 1),
                "backend": describe(),
            },
            indent=2,
        )
    )
    log.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
