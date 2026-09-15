"""Export a run to JSON for the browser scene.

    python scripts/export_web.py --run runs/long --out web/data.json

The browser does not need spikes at 0.1 ms resolution. It needs, per rendered
frame, five heat levels and six firing rates, which is three orders of magnitude
less data: a 30,000-step run becomes 900 frames at 30 fps.

Everything here is downsampling of recorded arrays. No simulation, no
reinterpretation, nothing the 2D figures do not already show.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from amfly.config import CHAMBERS, LIF, MEASURED, OPERATOR  # noqa: E402
from amfly.io.spikes import load_run  # noqa: E402

log = logging.getLogger("amfly.export")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=Path("runs/long"))
    ap.add_argument("--out", type=Path, default=Path("web/data.json"))
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=30.0,
                    help="playback length; the run is stretched or compressed "
                         "to fill it, since simulated time and clip length are "
                         "independent")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    data = load_run(args.run / "run.npz")
    heat = data["heat"]
    rates = data["rates"].astype(np.float64)
    ham = data.get("hamming")
    steps = len(heat)
    dt = LIF().dt_ms

    frames = int(round(args.fps * args.seconds))
    idx = np.linspace(0, steps - 1, frames).astype(np.int64)

    # Rates are spiky at single-step resolution; average over each frame's
    # span so the browser shows a rate rather than one arbitrary sample.
    edges = np.linspace(0, steps, frames + 1).astype(np.int64)
    rate_frames = np.stack(
        [rates[a:b].mean(axis=0) if b > a else rates[a] for a, b in
         zip(edges[:-1], edges[1:])]
    )

    amp = max(float(heat.max()), 1e-9)
    payload = {
        "meta": {
            "neurons": MEASURED.neurons,
            "connections": MEASURED.connections,
            "synapses": MEASURED.synapses,
            "instances": 6,
            "chambers": list(CHAMBERS),
            "operator": OPERATOR,
            "simulated_ms": steps * dt,
            "fps": args.fps,
            "frames": frames,
            "source_run": str(args.run),
        },
        # 0..1 per chamber per frame.
        "heat": np.round(heat[idx][:, list(CHAMBERS)] / amp, 4).tolist(),
        # Normalised 0..1 so the browser does not need the absolute scale.
        "rates": np.round(
            rate_frames / max(rate_frames.max(), 1.0), 4
        ).tolist(),
    }

    if ham is not None and len(ham):
        cum = np.cumsum(ham, axis=0)[idx]
        payload["divergence"] = np.round(
            cum / max(cum.max(), 1.0), 5
        ).tolist()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, separators=(",", ":")))
    size_kb = args.out.stat().st_size / 1024
    log.info(
        "wrote %s: %d frames, %.1f s at %d fps, %.0f KB",
        args.out, frames, args.seconds, args.fps, size_kb,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
