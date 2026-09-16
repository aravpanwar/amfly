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

    # Outgoing synapse count per DN, used to balance the blocks. None falls
    # back to contiguous slices, which are measurably unfair; see _partition.
    dn_weights: np.ndarray | None = None

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
    # 3.0 with the tracking update gives 75% of the run in motion and nothing
    # pinned. Under the old integrator no contrast value helped, because the
    # accumulation saturated regardless.
    contrast: float = 3.0

    # How many buttons the operator can press at once. None or n_blocks means
    # no limit, which is the original design.
    grip: int | None = None

    # Buttons rather than dials: pressing raises a level, releasing lets it
    # fall. Set False for the original tracking behaviour.
    buttons: bool = True

    # How strongly accumulated neglect pulls the operator towards a chamber it
    # has been ignoring. 0 restores the open loop.
    debt_bias: float = 1.1


    # A button is faster to raise than a released level is to fall, so holding
    # a chamber high is possible but only by staying on it.
    # Slower than the 1/260 and 1/900 used before, so the rise and fall are
    # themselves visible rather than instantaneous jumps between rails.
    press_rate: float = 1.0 / 900.0
    release_rate: float = 1.0 / 2200.0

    history: list = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._blocks = self._partition()
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
        # Set each step by the caller from Compulsion._debt, so the buttons can
        # see what the operator is being burned for.
        self._debt_view = np.zeros(self.n_blocks, dtype=np.float64)
        self._held = np.arange(min(2, self.n_blocks))

    def set_debt(self, debt) -> None:
        """Tell the buttons which chambers are owed attention."""
        self._debt_view = np.asarray(debt, dtype=np.float64)[: self.n_blocks]

    def _partition(self) -> list:
        """Split the DNs into blocks of comparable synaptic output.

        Contiguous slices of sorted bodyId are NOT comparable. Measured, the
        five blocks carried 1,586,011 / 1,076,376 / 486,094 / 236,733 / 707,345
        outgoing synapses, a 6.70x spread, because bodyId ordering correlates
        with neuron size and connectivity. Blocks 0 and 1 therefore fired more
        whatever the fly was doing, and their chambers sat above half heat for
        99% of a run while the other three sat at the floor for 96%.

        That is the anatomy deciding, not the operator. Dealing each DN to the
        currently lightest block (greedy longest-processing-time) brings the
        spread to 1.00x with the blocks still holding ~263 DNs each, so a
        difference between blocks now means the fly is doing something
        different rather than one block simply being bigger.

        Still a fixed, declared rule. No RNG, no training, and the partition is
        a pure function of the connectome.
        """
        if self.dn_weights is None:
            return np.array_split(self.dn_indices, self.n_blocks)

        w = np.asarray(self.dn_weights, dtype=np.float64)
        order = np.argsort(-w)          # heaviest first
        loads = np.zeros(self.n_blocks)
        groups = [[] for _ in range(self.n_blocks)]
        for j in order:
            k = int(np.argmin(loads))
            groups[k].append(int(self.dn_indices[j]))
            loads[k] += w[j]
        # Sort within a block so the partition is deterministic and auditable.
        return [np.array(sorted(g), dtype=np.int64) for g in groups]

    @classmethod
    def build(cls, dn_indices, dt_ms: float, window_ms: float = 50.0,
              dn_weights=None) -> "Dials":
        return cls(
            dn_indices=np.asarray(dn_indices),
            window_steps=max(1, int(round(window_ms / dt_ms))),
            latency_steps=max(1, int(round(DIAL_LATENCY_MS / dt_ms))),
            dn_weights=dn_weights,
        )

    @property
    def levels(self) -> np.ndarray:
        """(n_blocks,) heat level per chamber, each in 0..1."""
        return self._levels.copy()

    # Minimum steps between re-deciding which buttons are held.
    #
    # Without this the operator re-chose every step, 1720 changes per second,
    # while a press needs 26ms to raise a chamber. No press ever completed, so
    # the chambers that were not dominant just vibrated around mid-range and
    # the plot became a band of noise instead of legible strokes.
    #
    # This is not the old commitment, which held a slot until a chamber was
    # maxed and locked the quietest block out entirely. It only sets how often
    # the choice is revisited.
    # 300 steps, 30ms, gave 83 button transitions per second: about 250
    # strokes across a 3s plot, roughly 5px each and unreadable. A legible
    # clip wants 5 to 10 per second. 3000 steps is 300ms per decision, and
    # since a press lands in 26ms the chamber then simply sits held, which is
    # what "the operator is holding this one" should look like.
    # Measured across three runs, decisions per second and how it read:
    #   every step   1720/s  vibration, no press ever completed
    #   30ms           83/s  sawtooth, ~250 strokes at 5px each
    #   300ms          13/s  a metronome: perfectly regular rotation, and
    #                        motion fell to 26% with 67% spent at a rail
    #
    # At 300ms the cadence sets the rhythm rather than the operator's own
    # activity. 100ms is just longer than the ~90ms a press takes to land, so
    # a choice can complete, while leaving the brain enough say to break the
    # regularity.
    decide_every: int = 1000

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

        # BUTTONS, not dials.
        #
        # A dial holds a position when you let go. A button does not: press and
        # the level climbs, release and it falls. That is the more legible
        # object and the crueller one, because nothing the operator achieves
        # stays achieved. Holding a chamber at 100% means never stopping.
        #
        # GRIP: only the two strongest demands are actually pressed. The rest
        # are released and fall on their own.
        #
        # This is the constraint that makes the operator rush. With five dials
        # freely held it can satisfy everything and there is no conflict; with
        # two it must choose, and it is punished for the three it drops. The
        # limit is a rule of the piece, not a property of the fly, and the
        # README says so.
        if applied is not None and self.grip and self.grip < self.n_blocks:
            # Neglect debt biases WHICH buttons get pressed.
            #
            # The operator still chooses by its own activity; debt only tips
            # which win when they are close. That is what lets a persistently
            # quiet block still get attention: block 2 fires 11.5% below the
            # mean as a property of the connectome, so on raw activity alone it
            # would always be fifth of five. With debt it is held 21.9% of the
            # time and reaches about 40% heat.
            #
            # Select ONCE, here, and let the press below use this decision. An
            # earlier version ranked again in the press block using a mangled
            # copy of `applied`, which silently discarded the debt-aware choice
            # and made debt_bias do nothing at all.
            score = applied + self.debt_bias * self._debt_view
            # Re-decide on a cadence, not every step.
            if step % self.decide_every != 0:
                score = None
            if score is not None:
                self._held = np.argsort(-score)[: self.grip]

        if applied is not None and self.buttons:
            # Pressed climbs, released falls.
            #
            # Which buttons are pressed is decided by RANK, not by sign.
            #
            # The targets are mean-centred so they sum to about zero, which
            # means at most one or two can ever be positive and the rest are
            # pushed negative. Measured on real spikes the five targets were
            # -0.022, -0.024, -0.232, -0.045, +0.323: only one above zero.
            # Testing `applied > 0` therefore meant the consistently quietest
            # block could never press its button at all, whatever the grip,
            # bias or escalation. Chamber 2 read exactly 0.0 heat in every
            # configuration because of this line, not because of competition.
            #
            # Ranking instead lets the grip decide how many are held, which is
            # what the grip was always meant to control.
            if self.grip and self.grip < self.n_blocks:
                pressed = np.zeros(self.n_blocks, dtype=bool)
                pressed[self._held] = True
            else:
                pressed = applied > np.median(applied)
            delta = np.where(pressed, self.press_rate, -self.release_rate)
            self._levels += delta.astype(np.float32)
            np.clip(self._levels, 0.15, 1.0, out=self._levels)
        elif applied is not None:
            # Move TOWARDS the target, do not accumulate into it.
            #
            # This was `self._levels += applied * slew_rate`, an integrator with
            # no restoring force: a block even slightly above average pushed its
            # level up every step, and over 30,000 steps those increments always
            # reached a rail and stayed there. Measured, the five blocks fire
            # within 1.24x of each other, yet chambers 2 and 4 sat pinned for
            # 94-98% of a run. A 1.24x input was producing a 100%-vs-0% output.
            #
            # Mapping the target to a LEVEL and easing towards it means a
            # constant input holds a constant level, and only a change in the
            # operator's activity moves a dial.
            goal = 0.5 + applied * 0.5          # target in 0..1
            self._levels += ((goal - self._levels) * self.slew_rate * 40.0
                             ).astype(np.float32)
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
