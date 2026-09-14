"""Measure step time, so performance claims are measured rather than estimated.

    python scripts/benchmark.py --data data --steps 50

Reports ms per step and the implied wall clock for a 30-second clip. Run it
before and after installing a CUDA build of torch to see what the GPU actually
recovers on this machine, rather than trusting a number from elsewhere.

Expect wall-clock variance on a laptop. Thermal throttling shows up as step
time drifting upward over a long run. It does not affect correctness: the
arithmetic is deterministic regardless of clock speed.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from amfly.config import LIF  # noqa: E402
from amfly.data.loader import load  # noqa: E402
from amfly.sim.backend import describe  # noqa: E402
from amfly.sim.divergence import Divergence  # noqa: E402
from amfly.sim.engine import Engine, State  # noqa: E402
from amfly.wiring.chambers import Heat  # noqa: E402

log = logging.getLogger("amfly.bench")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    info = describe()
    log.info("backend: %s", info)

    t0 = time.time()
    c = load(args.data)
    log.info("load: %.1fs", time.time() - t0)

    lif = LIF()
    eng = Engine(c.csr, lif)
    st = State.initial(c.n, lif)
    heat = Heat(c.thermo_indices(), c.n, 6)
    div = Divergence()

    sensory = np.flatnonzero(
        np.char.startswith(c.superclass.astype(str), "cb_sensory")
    )
    drive = sensory[::7]

    def one(t: int) -> None:
        inj = heat.injection(0)
        if t % 100 < 5:
            inj[drive, :] += np.float32(5.0)
        s = eng.step(st, inj)
        div.update(s, t)

    for t in range(args.warmup):
        one(t)

    per_step = []
    for t in range(args.warmup, args.warmup + args.steps):
        a = time.perf_counter()
        one(t)
        per_step.append(time.perf_counter() - a)

    ms = np.array(per_step) * 1000.0
    log.info("")
    log.info("steps measured: %d", len(ms))
    log.info("  median  %7.1f ms", float(np.median(ms)))
    log.info("  mean    %7.1f ms", float(ms.mean()))
    log.info("  min     %7.1f ms", float(ms.min()))
    log.info("  max     %7.1f ms", float(ms.max()))

    # A drifting max against a low min on a laptop is usually thermal.
    if ms.max() > 1.5 * np.median(ms):
        log.info("  (spread suggests thermal throttling; wall clock only)")

    for label, seconds in (("1 s", 1.0), ("30 s clip", 30.0)):
        steps = seconds * 1000.0 / lif.dt_ms
        hours = float(np.median(ms)) * steps / 1000.0 / 3600.0
        log.info("projected for %-10s %6.1f hours", label, hours)

    if not info.get("cuda_available"):
        log.info("")
        log.info("CPU path. A CUDA build of torch is the single biggest win here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
