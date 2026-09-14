"""Turn a run dump into the figures and the clip.

    python scripts/make_figures.py --run runs/001 --out out/

Stills always. The mp4 and gif only when ffmpeg is present, and their absence
is reported rather than swallowed, because a silently missing clip is the one
output this project exists to produce.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from amfly.config import CHAMBERS, LIF, OPERATOR  # noqa: E402
from amfly.io.spikes import load_run  # noqa: E402
from amfly.sim.analysis import analyse, unheated_stayed_identical  # noqa: E402
from amfly.viz import traces  # noqa: E402

REFERENCE = CHAMBERS[1]  # an unheated chamber; see amfly/sim/divergence.py

log = logging.getLogger("amfly.figures")


def render_frames(data, out_dir: Path, n_frames: int, dt_ms: float) -> Path:
    """Progressive reveal: each frame shows the run up to that point.

    The clip is the traces being drawn, so the viewer watches them separate
    rather than being handed a finished plot.
    """
    frames = out_dir / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    T = len(data["rates"])
    for i in range(n_frames):
        upto = max(2, int(T * (i + 1) / n_frames))
        fig = traces.six_traces(
            data["rates"], data["heat"], data["dial"], dt_ms=dt_ms, upto=upto
        )
        traces.save(fig, frames / f"f{i:05d}.png", dpi=100)
        if i % 20 == 0:
            log.info("  frame %d/%d", i, n_frames)
    return frames


def encode(frames: Path, out: Path, fps: int) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        log.warning("ffmpeg not found; skipping mp4 and gif")
        return False

    mp4 = out / "amfly.mp4"
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps),
         "-i", str(frames / "f%05d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(mp4)],
        check=True,
    )
    log.info("wrote %s", mp4)

    # Palette pass, otherwise the gif dithers the dark background badly.
    palette = out / "palette.png"
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(mp4),
         "-vf", "fps=%d,scale=900:-1:flags=lanczos,palettegen" % fps,
         str(palette)],
        check=True,
    )
    gif = out / "amfly.gif"
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(mp4), "-i", str(palette),
         "-lavfi", "fps=%d,scale=900:-1:flags=lanczos [x]; [x][1:v] paletteuse" % fps,
         str(gif)],
        check=True,
    )
    palette.unlink(missing_ok=True)
    log.info("wrote %s", gif)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=Path("runs/latest"))
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--frames", type=int, default=180)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stills-only", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    data = load_run(args.run / "run.npz")
    rates = data["rates"]
    dt = LIF().dt_ms
    log.info("loaded %d steps (%.1f ms simulated)", len(rates), len(rates) * dt)

    args.out.mkdir(parents=True, exist_ok=True)

    fig = traces.six_traces(rates, data["heat"], data["dial"], dt_ms=dt)
    traces.save(fig, args.out / "traces.png")
    log.info("wrote %s", args.out / "traces.png")

    ham = data.get("hamming")
    if ham is not None and len(ham):
        cum = np.cumsum(ham, axis=0)
        fig = traces.divergence(cum, dt_ms=dt, reference=REFERENCE)
        traces.save(fig, args.out / "divergence.png")
        log.info("wrote %s", args.out / "divergence.png")

        # State the finding plainly either way. A run where nothing separated
        # is a real result and gets reported, not hidden.
        for i in range(ham.shape[1]):
            if i == REFERENCE:
                continue
            label = "operator" if i == OPERATOR else f"chamber {i}"
            log.info("%s: %s", label, analyse(ham, i).summary())

        # The property every plot from this run depends on.
        #
        # Which instances were heated comes from the recorded heat, not from
        # the divergence itself. Once the dial moves, several chambers are
        # heated over a run, and inferring a single heated chamber from the
        # distances would flag a legitimate multi-chamber run as drift.
        ever_heated = set(np.flatnonzero(data["heat"].max(axis=0) > 0).tolist())
        strays = [
            i
            for i in range(ham.shape[1])
            if i != REFERENCE and i not in ever_heated and np.any(ham[:, i] != 0)
        ]
        if not strays:
            log.info(
                "never-heated instances stayed bit-identical (%d of them)",
                ham.shape[1] - len(ever_heated) - 1,
            )
        else:
            log.error(
                "instances %s were never heated yet diverged. Something other "
                "than heat is separating them, so these plots do not mean what "
                "they appear to. Check determinism before using this run.",
                strays,
            )
    else:
        log.warning("no hamming data in run; divergence plot skipped")

    if not args.stills_only:
        frames = render_frames(data, args.out, args.frames, dt)
        encode(frames, args.out, args.fps)

    return 0


if __name__ == "__main__":
    sys.exit(main())
