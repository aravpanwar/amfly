"""Five dials, not one switch.

The operator does not choose which chamber is heated. It sets how much, for each
of the five, continuously. Every chamber is always being heated by some amount.
Nothing is ever off.

Each of the five descending-neuron blocks drives one chamber's level directly:
block 0 busy means chamber 0 climbs, block 0 quiet means chamber 0 falls. The
five move independently, so all five can be high at once, or all low, or any mix.

This replaces the single-selection dial in operator.py. Two reasons.

The first is the piece. A switch is a choice made once. Five levels held
continuously is an ongoing act with nothing ever off the hook, and the five
chambers are not taking turns, they are all being adjusted forever by something
that does not know they exist.

The second is measurement. Only a continuously heated chamber accumulated any
real divergence: chamber 0 reached a cumulative distance of 43.5 million over
13,637 uninterrupted steps, while chambers heated in blocks of 2,805 peaked at a
single neuron. See docs/negative-results.md. Under one switch, four chambers sit
cold at any moment and never accumulate. Under five dials they all do.

As before: the operator reads only its own spikes, never the chambers. It gets
no feedback. There is no RNG anywhere in this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import CHAMBERS, DIAL_LATENCY_MS, OPERATOR


@dataclass
class Dials:
    """Five continuous heat levels, driven by five blocks of descending neurons."""

    dn_indices: np.ndarray
    window_steps: int
    latency_steps: int
    n_blocks: int = len(CHAMBERS)

    # How fast a level can move. A level crosses its full range in roughly
    # 1/rate steps, so 1/2000 is about 200ms at dt=0.1ms. Slow enough that a
    # viewer can watch a chamber climb rather than jump.
    slew_rate: float = 1.0 / 2000.0

    # Each block is compared against its own slow baseline, so a persistently
    # quiet block can still drive its chamber up when it becomes unusually
    # active. Without this the blocks with the highest raw rates would pin
    # their chambers high forever: measured block totals were
    # [228, 231, 150, 152, 184]. See docs/negative-results.md.
    _baseline_decay: float = 0.999
    _baseline_floor: float = 1.0

    history: list = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._blocks = np.array_split(self.dn_indices, self.n_blocks)
        self._counts = np.zeros((self.window_steps, self.n_blocks), dtype=np.int32)
        self._baseline = np.zeros(self.n_blocks, dtype=np.float64)
        self._levels = np.zeros(self.n_blocks, dtype=np.float32)
        self._queue: list = []

    @classmethod
    def build(cls, dn_indices, dt_ms: float, window_ms: float = 50.0) -> "Dials":
        return cls(
            dn_indices=np.asarray(dn_indices),
            window_steps=max(1, int(round(window_ms / dt_ms))),
            latency_steps=max(1, int(round(DIAL_LATENCY_MS / dt_ms))),
        )

    @property
    def levels(self) -> np.ndarray:
        """(n_blocks,) heat level per chamber, each in 0..1."""
        return self._levels.copy()

    def update(self, spikes: np.ndarray, step: int) -> np.ndarray:
        """One step of operator spikes in, five heat levels out.

        `spikes` is the full (N, n_instances) matrix. Only the operator column
        is read.
        """
        operator_spikes = spikes[:, OPERATOR]

        slot = step % self.window_steps
        for b, block in enumerate(self._blocks):
            self._counts[slot, b] = int(operator_spikes[block].sum())

        totals = self._counts.sum(axis=0).astype(np.float64)

        # Compare each block against the average across blocks, NOT against its
        # own running baseline.
        #
        # Per-block baselines were the first attempt and they are wrong here.
        # A baseline that chases its own signal drives every block to the same
        # ratio: measured, blocks with raw totals 1000 and 500 both normalised
        # to exactly 1.161, so all five levels moved in lockstep and the five
        # dials were one dial. Self-normalisation removes exactly the
        # differences between blocks that these dials need.
        #
        # A shared reference keeps the comparison between blocks. The mean is
        # itself driven entirely by operator spikes, so there is still no RNG
        # and no authored constant deciding which chamber suffers.
        self._baseline *= self._baseline_decay
        self._baseline += (1.0 - self._baseline_decay) * totals.mean()
        reference = max(float(self._baseline.mean()), self._baseline_floor)

        # Above the shared reference pushes a chamber up, below pushes it down.
        # Clipped so one loud burst cannot slam a level to the rail.
        target = np.clip(totals / reference - 1.0, -1.0, 1.0)

        # The authored lag: what the operator's brain is doing now reaches the
        # chambers 500ms from now, so a viewer can learn to predict it.
        self._queue.append((step + self.latency_steps, target))
        applied = None
        while self._queue and self._queue[0][0] <= step:
            _, applied = self._queue.pop(0)

        if applied is not None:
            self._levels += (applied * self.slew_rate).astype(np.float32)
            np.clip(self._levels, 0.0, 1.0, out=self._levels)

        self.history.append(self._levels.copy())
        return self.levels

    def level_history(self) -> np.ndarray:
        """(T, n_blocks) levels over the run. Drives the dial panel."""
        return np.array(self.history, dtype=np.float32)

    def block_totals(self) -> np.ndarray:
        return self._counts.sum(axis=0)
