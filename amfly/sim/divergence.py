"""Measuring how far apart the instances have drifted.

Population rate is the wrong observable and the first run proved it: chamber 0
diverged in *which* neurons fired while the total count stayed identical, so a
rate trace showed six lines on top of each other and hid the entire subject of
the piece. See docs/negative-results.md.

What actually matters is spike identity. Two instances are the same only while
the same neurons fire at the same step.

`hamming` is the honest measure and it is cheap: one XOR over the boolean spike
matrix per step, no allocation of the full history.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import CHAMBERS


@dataclass
class Divergence:
    """Running spike-identity distance of every instance from a reference.

    Tracks both the per-step distance and its cumulative sum. The cumulative
    version is the one that reads on a plot: it is flat at zero while two
    instances are the same program, and it lifts and never returns once they
    are not.
    """

    reference: int = CHAMBERS[0]
    n_instances: int = 6
    per_step: list = field(default_factory=list, init=False)
    first_divergence: dict = field(default_factory=dict, init=False)

    def update(self, spikes: np.ndarray, step: int) -> np.ndarray:
        """spikes: (N, n_instances) bool. Returns (n_instances,) distance."""
        ref = spikes[:, self.reference]
        d = np.empty(self.n_instances, dtype=np.int32)
        for i in range(self.n_instances):
            if i == self.reference:
                d[i] = 0
                continue
            n = int(np.count_nonzero(spikes[:, i] ^ ref))
            d[i] = n
            if n and i not in self.first_divergence:
                self.first_divergence[i] = step
        self.per_step.append(d)
        return d

    def history(self) -> np.ndarray:
        """(T, n_instances) per-step Hamming distance from the reference."""
        return np.array(self.per_step, dtype=np.int32)

    def cumulative(self) -> np.ndarray:
        """(T, n_instances) cumulative distance. This is what gets plotted."""
        return np.cumsum(self.history(), axis=0)


def spike_identity_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Neurons that fired in exactly one of the two. Zero means same program."""
    return int(np.count_nonzero(a ^ b))
