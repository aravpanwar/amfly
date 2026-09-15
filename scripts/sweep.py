"""Overnight parameter sweep.

    python scripts/sweep.py --hours 7 --out runs/sweep

Runs the simulation across a grid of compulsion and button parameters, writing
one run per combination plus a summary table, so the morning is a matter of
picking a configuration rather than guessing at one.

Every parameter here was chosen by hand in a single evening, mostly by eye:
grip, debt_bias, contrast, habituation and escalation rates. None has been
explored. That is what this is for.

Results are appended to sweep.csv as each run finishes, so an interrupted
sweep still leaves usable data. The laptop may sleep, throttle or be closed;
none of that corrupts what is already written.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

log = logging.getLogger("amfly.sweep")

# The grid. Kept small deliberately: a 3s run is about 10 minutes, so seven
# hours is roughly 40 runs. Better to explore a few axes properly than to
# sample a large space once each.
GRID = {
    "grip": [2, 3],
    "debt-bias": [0.0, 1.1, 2.5],
    "contrast": [2.0, 3.0, 4.5],
    "escalate": [0.0009, 0.003],
}


def score(run_dir: Path) -> dict:
    """What makes a run good to watch, measured rather than eyeballed."""
    d = np.load(run_dir / "run.npz")
    h = d["heat"][:, :5]
    amp = max(float(h.max()), 1e-9)
    lv = h / amp

    # Spread: are the five doing different things at any given moment?
    spread = float(np.mean(lv.max(axis=1) - lv.min(axis=1)))
    # Motion: how much of the run is something actually changing?
    motion = float((np.abs(np.diff(lv, axis=0)).sum(axis=1) > 0.002).mean())
    # Coverage: does every chamber get real attention, or is one written off?
    reach = [float((lv[:, i] > 0.5).mean()) for i in range(5)]
    # Rails: time pinned at either extreme, which reads as a flat line.
    pinned = float(((lv > 0.97) | (lv < 0.18)).mean())

    prov = json.loads((run_dir / "provenance.json").read_text())
    comp = prov.get("compulsion") or {}
    return {
        "spread": round(spread, 4),
        "motion": round(motion, 4),
        "min_reach": round(min(reach), 4),
        "pinned": round(pinned, 4),
        "reward": round(comp.get("mean_reward", 0), 4),
        "punish": round(comp.get("mean_punish", 0), 4),
        "neglected": round(comp.get("mean_neglected_chambers", 0), 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path,
                    default=Path(r"C:\Users\ARVAPA~1\AppData\Local\Temp"))
    ap.add_argument("--out", type=Path, default=Path("runs/sweep"))
    ap.add_argument("--ms", type=float, default=3000.0)
    ap.add_argument("--hours", type=float, default=7.0)
    ap.add_argument("--record-sample", type=int, default=4000,
                    help="smaller than a hero run; the sweep is for choosing "
                         "parameters, not for the final brain view")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "sweep.csv"
    combos = list(itertools.product(*GRID.values()))
    keys = list(GRID.keys())

    log.info("%d combinations, budget %.1f h", len(combos), args.hours)
    deadline = time.time() + args.hours * 3600

    done = set()
    if csv_path.exists():
        with open(csv_path) as fh:
            for row in csv.DictReader(fh):
                done.add(row["name"])
        log.info("resuming: %d already complete", len(done))

    fresh = not csv_path.exists()
    with open(csv_path, "a", newline="") as fh:
        writer = None
        for i, combo in enumerate(combos):
            name = "_".join(f"{k}{v}" for k, v in zip(keys, combo))
            if name in done:
                continue
            if time.time() > deadline:
                log.info("budget spent, stopping with %d runs done", len(done))
                break

            run_dir = args.out / name
            cmd = [
                sys.executable, "scripts/run_sim.py",
                "--data", str(args.data), "--ms", str(args.ms),
                "--record-sample", str(args.record_sample),
                "--out", str(run_dir),
            ]
            for k, v in zip(keys, combo):
                cmd += [f"--{k}", str(v)]

            log.info("[%d/%d] %s", i + 1, len(combos), name)
            t0 = time.time()
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0 or not (run_dir / "run.npz").exists():
                log.error("  FAILED: %s", r.stderr.strip().splitlines()[-1:])
                continue

            row = {"name": name, "minutes": round((time.time() - t0) / 60, 1)}
            row.update(dict(zip(keys, combo)))
            row.update(score(run_dir))
            if writer is None:
                writer = csv.DictWriter(fh, fieldnames=list(row))
                if fresh:
                    writer.writeheader()
            writer.writerow(row)
            fh.flush()
            done.add(name)
            log.info("  %.1f min  spread %.3f  motion %.2f  min_reach %.2f",
                     row["minutes"], row["spread"], row["motion"],
                     row["min_reach"])

    log.info("sweep finished: %d runs in %s", len(done), csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
