"""The sixth fly, and the dial.

The operator is not special. Same graph, same weights, same parameters as the
five in the chambers. It differs only in where it sits in the wiring: its
descending-neuron activity is read out and used to point a dial, and the dial
selects which chamber gets heated.

It receives no feedback. It never sees the chambers, and nothing about their
state reaches it. Indifference rather than cruelty.

The readout is a fixed rule, not a trained decoder. Partition the 1,314
descending neurons into 5 contiguous blocks by sorted bodyId, count spikes per
block in a sliding window, point at the argmax. Ties break to the lowest index.
Nothing is learned, nothing is randomised, and the mapping can be audited by
reading this file.

The partition is arbitrary and is declared as such. Measured: the five blocks
hold 263, 263, 263, 263 and 262 DNs and span 181, 183, 142, 81 and 180 distinct
DN types respectively, so no block is a single functional group and the mapping
carries no anatomical claim. It is a legible rule for turning one fly's
descending activity into a choice of five, nothing more. Reading it as "block 3
is the escape pathway" would be wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import CHAMBERS, DIAL_LATENCY_MS, OPERATOR


@dataclass
class Dial:
    """Reads operator DN activity, points at a chamber, with a visible lag.

    The latency is deliberate and authored. It is long enough that a viewer
    learns to predict the chamber event from the spike burst before they
    understand the mechanism, which is what makes the silences do work.
    """

    dn_indices: np.ndarray
    window_steps: int
    latency_steps: int
    n_blocks: int = len(CHAMBERS)

    _counts: np.ndarray = field(init=False)
    _pending: list = field(default_factory=list, init=False)
    _position: int = field(default=0, init=False)
    history: list = field(default_factory=list, init=False)

    # Slow baseline per block, so the comparison is against each block's own
    # recent history rather than against the other blocks' absolute rates.
    _baseline_decay: float = 0.999
    _baseline_floor: float = 1.0

    def __post_init__(self) -> None:
        # Contiguous blocks over the DNs, already in sorted bodyId order.
        self._blocks = np.array_split(self.dn_indices, self.n_blocks)
        self._counts = np.zeros((self.window_steps, self.n_blocks), dtype=np.int32)
        self._baseline = np.zeros(self.n_blocks, dtype=np.float64)

    @classmethod
    def build(cls, dn_indices, dt_ms: float, window_ms: float = 50.0) -> "Dial":
        return cls(
            dn_indices=np.asarray(dn_indices),
            window_steps=max(1, int(round(window_ms / dt_ms))),
            latency_steps=max(1, int(round(DIAL_LATENCY_MS / dt_ms))),
        )

    @property
    def position(self) -> int:
        """Chamber currently selected."""
        return self._position

    def update(self, spikes: np.ndarray, step: int) -> int:
        """Feed one step of spikes, return the dial position now in effect.

        `spikes` is the full (N, n_instances) matrix. Only the operator column
        is read; the chamber columns are never consulted.
        """
        operator_spikes = spikes[:, OPERATOR]

        slot = step % self.window_steps
        for b, block in enumerate(self._blocks):
            self._counts[slot, b] = int(operator_spikes[block].sum())

        totals = self._counts.sum(axis=0).astype(np.float64)

        # Normalise each block by its own slow baseline before comparing.
        #
        # Raw argmax over these counts has a fixed winner: measured block totals
        # were [228, 231, 150, 152, 184], so blocks 0 and 1 always won and three
        # chambers were never heated. Equal-sized blocks equalise neuron count,
        # not firing rate, and DN rates are not uniform across sorted bodyId.
        # See docs/negative-results.md.
        #
        # Dividing by the running mean asks "which block is unusually active
        # right now" instead of "which block is loudest", which is both the more
        # interesting question and the one that can actually select all five.
        # Still driven entirely by spikes, with no RNG.
        self._baseline *= self._baseline_decay
        self._baseline += (1.0 - self._baseline_decay) * totals

        relative = totals / np.maximum(self._baseline, self._baseline_floor)
        # argmax breaks ties to the lowest index, deterministically.
        target = int(np.argmax(relative))

        # Commit the decision to fire `latency_steps` from now.
        self._pending.append((step + self.latency_steps, target))

        while self._pending and self._pending[0][0] <= step:
            _, pos = self._pending.pop(0)
            if pos != self._position:
                self.history.append((step, self._position, pos))
            self._position = pos

        return self._position

    def block_totals(self) -> np.ndarray:
        """Per-block spike totals in the current window. For the dial readout."""
        return self._counts.sum(axis=0)
