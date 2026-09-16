"""Reading a finished run.

The question this answers is the one the plan flagged as a risk: does the heated
chamber stay separated, or does it fall back into step with the others?

Re-convergence would be a real finding rather than a bug. A deterministic
network settling back onto the same trajectory after a perturbation is a
statement about the connectome, and per the README register it gets published
either way rather than quietly reframed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Verdict:
    """What a run actually showed."""

    diverged: bool
    onset_step: int | None
    reconverged: bool
    reconvergence_step: int | None
    final_distance: int
    peak_distance: int
    silent_fraction: float

    def summary(self) -> str:
        if not self.diverged:
            return "never diverged; still the same program"
        if self.reconverged:
            return (
                f"diverged at step {self.onset_step}, then re-converged at step "
                f"{self.reconvergence_step}. Publish this as a negative result."
            )
        return (
            f"diverged at step {self.onset_step} and stayed separated "
            f"(final {self.final_distance}, peak {self.peak_distance})"
        )


def analyse(hamming: np.ndarray, instance: int, hold_steps: int = 200) -> Verdict:
    """Classify one instance's trajectory.

    `hamming` is (T, n_instances) per-step spike-identity distance from the
    reference. `hold_steps` is how long the distance must sit at zero before it
    counts as re-convergence rather than a momentary coincidence: two
    trajectories can briefly agree without having merged.
    """
    d = np.asarray(hamming)[:, instance]
    nonzero = np.flatnonzero(d)

    if len(nonzero) == 0:
        return Verdict(
            diverged=False,
            onset_step=None,
            reconverged=False,
            reconvergence_step=None,
            final_distance=0,
            peak_distance=0,
            silent_fraction=1.0,
        )

    onset = int(nonzero[0])

    # Re-convergence: a run of zeros at least hold_steps long, after onset,
    # that reaches the end of the record.
    reconv_step = None
    tail = d[onset:]
    zeros = tail == 0
    if zeros[-1] and len(tail) >= hold_steps:
        # Walk back to the start of the final zero run.
        i = len(tail) - 1
        while i >= 0 and zeros[i]:
            i -= 1
        run_len = len(tail) - 1 - i
        if run_len >= hold_steps:
            reconv_step = onset + i + 1

    return Verdict(
        diverged=True,
        onset_step=onset,
        reconverged=reconv_step is not None,
        reconvergence_step=reconv_step,
        final_distance=int(d[-1]),
        peak_distance=int(d.max()),
        silent_fraction=float(np.mean(d == 0)),
    )


def unheated_stayed_identical(
    hamming: np.ndarray, heated: int, reference: int
) -> bool:
    """The property the whole piece rests on.

    Every instance except the heated one and the reference must sit at exactly
    zero for the entire run. If this is False the instances are drifting for
    some reason other than heat, and no plot from the run means anything.
    """
    h = np.asarray(hamming)
    for i in range(h.shape[1]):
        if i in (heated, reference):
            continue
        if np.any(h[:, i] != 0):
            return False
    return True
