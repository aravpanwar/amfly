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
from amfly.sim.analysis import analyse  # noqa: E402
from amfly.viz import traces  # noqa: E402

REFERENCE = OPERATOR  # never heated; see amfly/sim/divergence.py

log = logging.getLogger("amfly.figures")


def render_frames(data, out_dir: Path, n_frames: int, dt_ms: float,
                  which: str = "dials") -> Path:
    """Progressive reveal: each frame shows the run up to that point.

    Defaults to the dial plot rather than the six rate traces. The dials are
    what separate visibly: every chamber is heated, so cumulative divergence
    from the unheated operator bunches all five within 1.17x, while the heat
    levels themselves read as five distinct lines.
    """
    frames = out_dir / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    T = len(data["rates"])
    heat = data["heat"]
    for i in range(n_frames):
        upto = max(2, int(T * (i + 1) / n_frames))
        if which == "dials":
            # Hold the full x range so the line grows into a fixed frame
            # instead of the axis rescaling under it every step.
            fig = traces.dials(heat[:upto], dt_ms=dt_ms, xmax_ms=T * dt_ms)
        else:
            fig = traces.six_traces(
                data["rates"], heat, data["dial"], dt_ms=dt_ms, upto=upto
            )
        traces.save(fig, frames / f"f{i:05d}.png", dpi=100)
        if i % 40 == 0:
            log.info("  frame %d/%d", i, n_frames)
    return frames


def encode(frames: Path, out: Path, fps: int, gif_width: int = 720) -> bool:
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
         "-vf", "fps=%d,scale=%d:-1:flags=lanczos,palettegen" % (fps, gif_width),
         str(palette)],
        check=True,
    )
    gif = out / "amfly.gif"
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(mp4), "-i", str(palette),
         "-lavfi", "fps=%d,scale=%d:-1:flags=lanczos [x]; [x][1:v] paletteuse" % (fps, gif_width),
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
    ap.add_argument("--animate", default="dials", choices=["dials", "traces"],
                    help="which figure the clip animates")
    ap.add_argument("--gif-width", type=int, default=720,
                    help="gif width in px; 48MB at 900px is too big to post")
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

    fig = traces.dials(data["heat"], dt_ms=dt)
    traces.save(fig, args.out / "dials.png")
    log.info("wrote %s", args.out / "dials.png")

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
        # Report what each chamber actually got, not just time above a
        # threshold. A "time above 80%" metric read 0% for a chamber that was
        # in fact being pressed 21.9% of the time and sitting around 40% heat,
        # and sent an evening into chasing a stranding that was not happening.
        hh = data["heat"][:, : len(CHAMBERS)]
        amp = max(float(hh.max()), 1e-9)
        log.info("chamber heat: mean / peak / %% of run above half")
        for ci in range(hh.shape[1]):
            log.info(
                "  chamber %d: mean %3.0f%%  peak %3.0f%%  above-half %3.0f%%",
                ci, hh[:, ci].mean() / amp * 100, hh[:, ci].max() / amp * 100,
                (hh[:, ci] > amp * 0.5).mean() * 100,
            )

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
        frames = render_frames(data, args.out, args.frames, dt, args.animate)
        encode(frames, args.out, args.fps, args.gif_width)

    return 0


if __name__ == "__main__":
    sys.exit(main())
