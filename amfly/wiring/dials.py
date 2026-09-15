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
    # 1/rate steps, so 1/400 is about 40ms at dt=0.1ms.
    #
    # Was 1/2000. That took a full second for the five to separate, and the
    # chambers only diverge while their heat differs, so the slow opening was
    # dead footage. Faster separation gives the clip five distinguishable
    # traces within the first tenth of a second.
    slew_rate: float = 1.0 / 400.0

    # Each block is compared against its own slow baseline, so a persistently
    # quiet block can still drive its chamber up when it becomes unusually
    # active. Without this the blocks with the highest raw rates would pin
    # their chambers high forever: measured block totals were
    # [228, 231, 150, 152, 184]. See docs/negative-results.md.
    _baseline_decay: float = 0.999
    _baseline_floor: float = 1.0

    # Multiplies the spread between blocks. The raw differences between DN
    # block rates are small relative to their common level, so without this the
    # five dials sit within about 1.8mV of each other and the chambers cannot
    # tell them apart.
    # Was 8.0, which slammed the dials to the rails: measured, chamber 1 sat at
    # maximum 85% of a run and chamber 4 at zero 100% of it, so the plot showed
    # two overlapping lines and three flat ones. 2.5 keeps the five spread
    # across the middle of the range where they stay distinguishable.
    contrast: float = 2.5

    history: list = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._blocks = np.array_split(self.dn_indices, self.n_blocks)
        self._counts = np.zeros((self.window_steps, self.n_blocks), dtype=np.int32)
        self._baseline = np.zeros(self.n_blocks, dtype=np.float64)
        # Start at zero, NOT mid-range.
        #
        # Starting all five at 0.5 was an attempt to let levels move both ways
        # from the outset. It silently broke the piece: every chamber then
        # begins identically heated, the network locks onto a shared
        # trajectory, and it never escapes. Measured, five chambers that later
        # sat at 8.0, 8.0, 0.6, 0.0 and 0.0 mV still produced identical spike
        # trains, distance 1678 for all five, while the same final levels held
        # constant from step 0 gave 1663, 1663, 1342, 0, 0.
        #
        # An abrupt switch-on partway through is worse still: heat applied from
        # step 1000 produced distance 0 everywhere. What matters is that the
        # chambers differ from the very first step, and from zero they do,
        # because each rises at its own rate. See docs/negative-results.md.
        # Seed each chamber at a DIFFERENT level, spread across the range.
        #
        # A shared starting value, even a nonzero one, is the identical-start
        # trap: measured, all five starting together at 1.2 mV produced peak
        # 2161 for every chamber, identical, because they lock onto one
        # trajectory before the dials separate. They must differ at step 0.
        #
        # The spread here is authored and must be declared in the README. What
        # the operator controls is where each dial GOES from its seed, which is
        # still entirely spike-driven.
        self._levels = np.linspace(
            0.15, 0.85, self.n_blocks, dtype=np.float32
        )
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
        #
        # `contrast` scales the differences between blocks. Without it the term
        # below is dominated by a common drift: measured, the five levels rose
        # together and reached a spread of only 1.8mV between hottest and
        # coldest, while a spread of 8mV is needed before the chambers become
        # distinguishable from each other. See docs/negative-results.md.
        #
        # Subtracting the mean of the ratios removes exactly that common drift,
        # leaving only how each block compares with the others. It is still
        # entirely spike-driven.
        ratio = totals / reference
        target = np.clip((ratio - ratio.mean()) * self.contrast, -1.0, 1.0)

        # The authored lag: what the operator's brain is doing now reaches the
        # chambers 500ms from now, so a viewer can learn to predict it.
        self._queue.append((step + self.latency_steps, target))
        applied = None
        while self._queue and self._queue[0][0] <= step:
            _, applied = self._queue.pop(0)

        if applied is not None:
            self._levels += (applied * self.slew_rate).astype(np.float32)
            # Floor at 0.15 rather than 0. The premise is that nothing is ever
            # off, and a chamber pinned at absolute zero is both off and a flat
            # line on the plot.
            np.clip(self._levels, 0.15, 1.0, out=self._levels)

        self.history.append(self._levels.copy())
        return self.levels

    def level_history(self) -> np.ndarray:
        """(T, n_blocks) levels over the run. Drives the dial panel."""
        return np.array(self.history, dtype=np.float32)

    def block_totals(self) -> np.ndarray:
        return self._counts.sum(axis=0)
