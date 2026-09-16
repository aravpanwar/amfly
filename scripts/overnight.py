"""Hero run, then a parameter sweep, with a hard deadline.

    python scripts/overnight.py --until 06:45

Two jobs in one unattended session. The hero run goes first because it is the
one that definitely improves things: the scene currently plays 3 simulated
seconds stretched over 30 wall-clock seconds, so any recording repeats. A 30
second run removes the loop.

The sweep takes whatever time is left. It writes its CSV as each combination
finishes and is ordered outward from the best known configuration, so being cut
off by the deadline still leaves the useful neighbourhood explored.

Nothing here runs past the deadline. Each job is checked against the remaining
time before it starts, and the sweep is given an explicit budget rather than
being killed partway through a run and leaving a half written directory.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

log = logging.getLogger("amfly.overnight")

# Measured on this machine: 20 ms/step on the RTX 4050, so 10,000 steps per
# simulated second is about 3.3 minutes. 1.35x of that covers the load, the
# export and the npz write, which is not free at this size.
MIN_PER_SIM_SECOND = 3.4


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path,
                    default=Path(r"C:\Users\ARVAPA~1\AppData\Local\Temp"))
    ap.add_argument("--until", default="06:45",
                    help="hard stop, HH:MM. Nothing starts that cannot finish")
    ap.add_argument("--hero-ms", type=float, default=30000.0)
    ap.add_argument("--hero-out", type=Path, default=Path("runs/hero"))
    ap.add_argument("--sweep-out", type=Path, default=Path("runs/sweep"))
    ap.add_argument("--skip-hero", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
    )

    now = datetime.now()
    hh, mm = (int(x) for x in args.until.split(":"))
    deadline = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if deadline <= now:
        deadline += timedelta(days=1)
    log.info("deadline %s, %.1f hours from now",
             deadline.strftime("%H:%M"), (deadline - now).total_seconds() / 3600)

    def hours_left() -> float:
        return (deadline - datetime.now()).total_seconds() / 3600

    # ---- hero run
    if not args.skip_hero:
        need = args.hero_ms / 1000 * MIN_PER_SIM_SECOND / 60
        if need > hours_left():
            log.warning("hero run needs %.1f h and %.1f h remain; skipping",
                        need, hours_left())
        else:
            log.info("hero run: %.0f ms simulated, estimated %.1f h",
                     args.hero_ms, need)
            t0 = time.time()
            r = subprocess.run([
                sys.executable, "scripts/run_sim.py",
                "--data", str(args.data),
                "--ms", str(args.hero_ms),
                "--record-sample", "20000",
                "--out", str(args.hero_out),
            ], capture_output=True, text=True)
            mins = (time.time() - t0) / 60
            if r.returncode != 0 or not (args.hero_out / "run.npz").exists():
                log.error("hero run FAILED after %.0f min: %s",
                          mins, r.stderr.strip().splitlines()[-3:])
            else:
                log.info("hero run done in %.0f min -> %s", mins, args.hero_out)
                # Export straight away. A finished run that was never exported
                # is a directory nobody looks at.
                for cmd, what in [
                    ([sys.executable, "scripts/export_web.py",
                      "--run", str(args.hero_out), "--out", "web/data.json",
                      "--seconds", "30"], "web data"),
                    ([sys.executable, "scripts/export_brain.py",
                      "--run", str(args.hero_out), "--points", "24000"],
                     "brain points"),
                ]:
                    e = subprocess.run(cmd, capture_output=True, text=True)
                    if e.returncode == 0:
                        log.info("exported %s", what)
                    else:
                        log.error("export of %s failed: %s", what,
                                  e.stderr.strip().splitlines()[-2:])

    # ---- sweep with whatever is left
    budget = hours_left() - 0.15      # leave a margin so nothing overruns
    if budget < 0.4:
        log.info("%.1f h left, not enough for a sweep run; stopping", budget)
        return 0

    log.info("sweep: %.1f h budget, about %d combinations",
             budget, int(budget * 60 / 14))
    r = subprocess.run([
        sys.executable, "scripts/sweep.py",
        "--data", str(args.data),
        "--out", str(args.sweep_out),
        "--hours", f"{budget:.2f}",
    ], text=True)
    log.info("sweep exited %d", r.returncode)
    log.info("finished with %.1f h to spare", hours_left())
    return 0


if __name__ == "__main__":
    sys.exit(main())
