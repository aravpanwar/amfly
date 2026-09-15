"""Export soma positions and per-neuron activity for the browser brain view.

    python scripts/export_brain.py --run runs/long --out web/brain.bin

This is the technique every other project uses: one point per cell body,
rendered as THREE.Points, with a colour buffer rewritten each frame from an
activity array. Nobody renders neuron shapes; the arbors would be millions of
segments and would not read as anything at a glance.

Soma coordinates come from the public neuPrint API (see data/soma.npz), not
from the 12.7GB syn-points download, which is not needed just to place a dot
per neuron.

Two files are written:
  brain.bin   positions (float32 xyz) then per-frame activity (uint8)
  brain.json  the metadata needed to read it

Binary rather than JSON because per-frame activity for tens of thousands of
neurons is where the size actually goes.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from amfly.config import LIF, N_INSTANCES  # noqa: E402
from amfly.data.loader import load  # noqa: E402
from amfly.io.spikes import load_run  # noqa: E402

log = logging.getLogger("amfly.brain")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=Path("runs/long"))
    ap.add_argument("--data", type=Path,
                    default=Path(r"C:\Users\ARVAPA~1\AppData\Local\Temp"))
    ap.add_argument("--soma", type=Path, default=Path("data/soma.npz"))
    ap.add_argument("--out", type=Path, default=Path("web/brain.bin"))
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--points", type=int, default=24000,
                    help="neurons to render per fly; the full 141k is more "
                         "than a browser needs and hides structure in a haze")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    soma = np.load(args.soma)
    soma_ids, soma_xyz = soma["body_ids"], soma["xyz"]
    log.info("somas available: %d", len(soma_ids))

    c = load(args.data)
    run = load_run(args.run / "run.npz")

    # Only neurons that are BOTH in the connectome and have a soma position.
    order = np.argsort(soma_ids)
    pos_in_soma = np.searchsorted(soma_ids, c.body_ids, sorter=order)
    pos_in_soma = np.clip(pos_in_soma, 0, len(soma_ids) - 1)
    matched = soma_ids[order[pos_in_soma]] == c.body_ids
    idx_all = np.flatnonzero(matched)
    log.info("connectome neurons with a soma: %d of %d", len(idx_all), c.n)

    # Subsample by stride, not randomly: no RNG anywhere in this project.
    stride = max(1, len(idx_all) // args.points)
    idx = idx_all[::stride][: args.points]
    xyz = soma_xyz[order[pos_in_soma[idx]]]

    # Centre and scale to roughly unit size; the browser scales from there.
    xyz = xyz - xyz.mean(axis=0)
    xyz = xyz / np.abs(xyz).max()
    log.info("rendering %d points per fly", len(idx))

    # Per-frame activity. The run recorded a 3,330-neuron subset at full
    # resolution; every other neuron has no per-neuron record, so its point
    # stays dark. That is stated in the README rather than faked.
    frames = int(round(args.fps * args.seconds))
    steps = len(run["rates"])
    edges = np.linspace(0, steps, frames + 1).astype(np.int64)

    sub = run["subset"]
    ev_step, ev_neuron, ev_inst = (
        run["events_step"], run["events_neuron"], run["events_instance"]
    )

    # Map recorded neurons onto our point indices.
    point_of = {int(v): k for k, v in enumerate(idx)}
    recorded = np.array([point_of.get(int(n), -1) for n in sub], dtype=np.int64)
    keep = recorded >= 0
    log.info("recorded neurons that are also rendered points: %d", int(keep.sum()))

    neuron_to_point = {int(sub[i]): int(recorded[i]) for i in np.flatnonzero(keep)}

    # Store activity ONLY for points that actually have a record.
    #
    # A dense (frames, instances, points) array is 130MB and 98% zeros, because
    # only the recorded subset has per-neuron spikes. Keep the full cloud for
    # structure and carry activity as a short list of live points plus an index
    # back into the cloud.
    live_points = sorted(set(neuron_to_point.values()))
    live_index = {p: k for k, p in enumerate(live_points)}
    log.info("points with activity: %d of %d", len(live_points), len(idx))

    act = np.zeros((frames, N_INSTANCES, len(live_points)), dtype=np.uint8)
    for f in range(frames):
        a, b = edges[f], edges[f + 1]
        m = (ev_step >= a) & (ev_step < b)
        if not m.any():
            continue
        for n, inst in zip(ev_neuron[m], ev_inst[m]):
            p = neuron_to_point.get(int(n))
            if p is not None:
                k = live_index[p]
                v = act[f, int(inst), k]
                if v < 251:
                    act[f, int(inst), k] = v + 64
        if f % 200 == 0:
            log.info("  frame %d/%d", f, frames)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    live_arr = np.array(live_points, dtype=np.uint32)
    with open(args.out, "wb") as fh:
        fh.write(xyz.astype(np.float32).tobytes())
        fh.write(live_arr.tobytes())
        fh.write(act.tobytes())

    meta = {
        "points": int(len(idx)),
        "live": int(len(live_points)),
        "frames": frames,
        "instances": N_INSTANCES,
        "fps": args.fps,
        "position_bytes": int(len(idx) * 3 * 4),
        "live_bytes": int(len(live_points) * 4),
        "note": (
            "Soma positions from the public neuPrint API. Only the recorded "
            "subset has per-neuron activity; other points stay dark."
        ),
    }
    Path(str(args.out).replace(".bin", ".json")).write_text(json.dumps(meta))
    size_mb = args.out.stat().st_size / 1e6
    log.info("wrote %s: %.1f MB (%d points x %d frames x %d instances)",
             args.out, size_mb, len(idx), frames, N_INSTANCES)
    return 0


if __name__ == "__main__":
    sys.exit(main())
