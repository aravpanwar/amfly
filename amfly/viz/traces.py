"""Six traces, equal visual weight.

project.md is right that this is the whole thing: if the divergence does not
read on a plot, no amount of rendering saves it. So the trace panels are the
deliverable, not a debug view, and they get the same weight as any chamber
rendering.

Layout rule, taken from project.md: the viewer must be able to read causality.
Traces spike, then the chamber changes. The dial band sits under the traces on
a shared x axis so the lag is visible rather than described.

No 3D, no browser, no physics engine. Matplotlib to frames to mp4.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..config import CHAMBERS, OPERATOR  # noqa: E402

# Chambers share one colour; the operator is the odd one out. The five are
# meant to read as interchangeable, because they are.
CHAMBER_COLOR = "#c8c8c8"
HEATED_COLOR = "#d94a3d"
OPERATOR_COLOR = "#7aa7d9"
BG = "#0d0d0d"
FG = "#e8e8e8"


def _style(ax) -> None:
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color("#333333")
    ax.tick_params(colors="#888888", labelsize=8)


def six_traces(
    rates: np.ndarray,
    heat: np.ndarray,
    dial: np.ndarray,
    dt_ms: float = 0.1,
    upto: int | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Six stacked rate traces plus the dial band.

    rates: (T, 6) spikes per step
    heat:  (T, 6) heat level per instance
    dial:  (T,)   selected chamber
    """
    T = len(rates) if upto is None else upto
    t = np.arange(T) * dt_ms

    fig, axes = plt.subplots(
        7, 1, figsize=(10, 9), sharex=True,
        gridspec_kw={"height_ratios": [1] * 6 + [0.45], "hspace": 0.12},
    )
    fig.patch.set_facecolor(BG)

    ymax = float(rates[:T].max()) * 1.1 if T else 1.0

    for row, inst in enumerate(list(CHAMBERS) + [OPERATOR]):
        ax = axes[row]
        _style(ax)
        is_operator = inst == OPERATOR
        color = OPERATOR_COLOR if is_operator else CHAMBER_COLOR

        ax.plot(t, rates[:T, inst], lw=0.7, color=color)

        if not is_operator:
            # Shade while this chamber is being heated.
            hot = heat[:T, inst] > 0.01
            if hot.any():
                ax.fill_between(
                    t, 0, ymax, where=hot, color=HEATED_COLOR,
                    alpha=0.16, linewidth=0,
                )
        ax.set_ylim(0, ymax)
        ax.set_ylabel(
            "operator" if is_operator else f"chamber {inst}",
            color="#999999", fontsize=8, rotation=0,
            ha="right", va="center", labelpad=28,
        )
        ax.set_yticks([])

    band = axes[6]
    _style(band)
    band.step(t, dial[:T], where="post", lw=1.0, color=HEATED_COLOR)
    band.set_ylim(-0.5, len(CHAMBERS) - 0.5)
    band.set_yticks(list(CHAMBERS))
    band.set_ylabel("dial", color="#999999", fontsize=8, rotation=0,
                    ha="right", va="center", labelpad=28)
    band.set_xlabel("ms", color="#888888", fontsize=8)

    if title:
        fig.suptitle(title, color=FG, fontsize=10, y=0.95)
    return fig


def divergence(
    cumulative: np.ndarray, dt_ms: float = 0.1, reference: int = OPERATOR
) -> plt.Figure:
    """When the chambers stop being the same program.

    Takes cumulative spike-identity distance from amfly.sim.divergence, not a
    rate difference. Rate hides the thing entirely: the first run diverged in
    which neurons fired while totals stayed identical. See
    docs/negative-results.md.

    Flat at zero means still the same program. The moment a line lifts is the
    moment that chamber stopped being interchangeable, and it never returns.
    """
    fig, ax = plt.subplots(figsize=(10, 3.2))
    fig.patch.set_facecolor(BG)
    _style(ax)

    t = np.arange(len(cumulative)) * dt_ms
    # All five chambers are plotted. The reference is the operator, which is
    # never heated, so no chamber has to be dropped from the plot.
    #
    # Chambers that never separated sit exactly on zero and overlap each other
    # invisibly, so a plain legend claims five distinguishable lines where the
    # eye can find one. Label the flat ones as a single entry and say so.
    # Threshold relative to the largest mover, not against zero. A chamber that
    # reached a cumulative distance of 1 while another reached 12,500 is flat
    # on this plot whatever the strict inequality says.
    peak = max(
        (cumulative[:, c].max() for c in CHAMBERS if c != reference), default=0
    )
    cutoff = max(1.0, peak * 0.01)

    flat, moved = [], []
    for c in CHAMBERS:
        if c == reference:
            continue
        (moved if cumulative[:, c].max() >= cutoff else flat).append(c)

    for c in moved:
        ax.plot(t, cumulative[:, c], lw=1.1, label=f"chamber {c}")
    for i, c in enumerate(flat):
        ax.plot(
            t, cumulative[:, c], lw=0.9, color="#5a5a5a",
            label=("unchanged: " + ", ".join(str(x) for x in flat)) if i == 0 else None,
        )

    ax.set_xlabel("ms", color="#888888", fontsize=8)
    ref_label = "operator" if reference == OPERATOR else f"chamber {reference}"
    ax.set_ylabel(
        f"cumulative spike-identity distance\nfrom the {ref_label}",
        color="#999999", fontsize=8,
    )
    leg = ax.legend(frameon=False, fontsize=7, ncol=4)
    for txt in leg.get_texts():
        txt.set_color("#999999")
    fig.tight_layout()
    return fig


def save(fig: plt.Figure, path: Path, dpi: int = 130) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return path


def dials(heat: np.ndarray, dt_ms: float = 0.1,
          amplitude_mv: float = 8.0, xmax_ms: float | None = None) -> plt.Figure:
    """The five dials over time. This is the picture the piece is about.

    Cumulative divergence-from-the-operator bunches: every chamber is heated, so
    every chamber differs from an unheated reference by a similar amount, and
    measured the spread between highest and lowest was only 1.17x. The heat
    levels themselves separate cleanly, and they are what the operator is
    actually doing.
    """
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor(BG)
    _style(ax)

    t = np.arange(len(heat)) * dt_ms
    # A colour belongs to a CHAMBER, not to its rank.
    #
    # This used to sort by the heat at the last frame. In a still that merely
    # looks tidy; in the animated version it is a real bug, because each frame
    # re-sorts on its own final value and the five lines therefore swap
    # colours as they cross. Measured across a 180-frame reveal the order
    # changed at almost every sampled frame, so no line could be followed for
    # more than a second. Since the buttons started switching on a margin the
    # chambers cross more often, which made it worse.
    #
    # Fixed colours also mean the same chamber is the same colour in the plot,
    # in the browser scene and between runs.
    colors = ["#d94a3d", "#e07a3f", "#c9a227", "#6f9e4c", "#4a7fb5"]
    for c in range(min(heat.shape[1], len(CHAMBERS))):
        ax.plot(t, heat[:, c] / amplitude_mv * 100.0, lw=1.6,
                color=colors[c % len(colors)], label=f"chamber {c}")

    # Headroom above 100 so the legend never sits on top of a pinned line.
    if xmax_ms is not None:
        ax.set_xlim(0, xmax_ms)
    ax.set_ylim(0, 128)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("ms", color="#888888", fontsize=9)
    ax.set_ylabel("heat level  (%)", color="#999999", fontsize=9)
    leg = ax.legend(frameon=False, fontsize=8, ncol=5, loc="upper left")
    for txt in leg.get_texts():
        txt.set_color("#999999")
    fig.tight_layout()
    return fig
