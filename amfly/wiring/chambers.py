"""Five sealed chambers, and the heat.

Heat is delivered as sustained depolarising current into the thermoreceptor
neurons of exactly one chamber at a time, matching the sustained firing of fly
warm cells above about 25C.

The isolation property lives here and is enforced structurally: `injection()`
writes into exactly one column of the (N, n_instances) matrix. The operator
column is never written. There is no code path by which heating chamber 3 can
touch chamber 1, which is what makes Gate 2 a real test rather than a hopeful
assertion.

Targets are TRN_VP* (25 bodies). There are no neurons of type "AC" in this
dataset despite the literature naming anterior cell thermoreceptors, so an AC
lookup returns nothing at all and would leave the heat channel silently dead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import CHAMBERS, OPERATOR


@dataclass
class Heat:
    """Ramped, sustained current into one chamber's thermoreceptors."""

    target_indices: np.ndarray  # TRN_VP* rows
    n_neurons: int
    n_instances: int
    amplitude_mv: float = 8.0
    # 20ms at dt=0.1ms. Was 200ms, which is longer than a short run, so the
    # stimulus spent the whole window ramping and never actually arrived.
    # See docs/negative-results.md.
    ramp_steps: int = 200

    def __post_init__(self) -> None:
        if len(self.target_indices) == 0:
            raise ValueError(
                "no thermosensory targets resolved. Check the TRN_VP prefix; "
                "note that type 'AC' does not exist in MaleCNS v1.0."
            )
        self._level = np.zeros(self.n_instances, dtype=np.float32)
        self._buf = np.zeros((self.n_neurons, self.n_instances), dtype=np.float32)
        self._step_up = np.float32(self.amplitude_mv / self.ramp_steps)

    def injection(self, dial_position: int) -> np.ndarray:
        """Return the (N, n_instances) injection for this step.

        Heat ramps up in the selected chamber and decays in the others, so the
        viewer sees a chamber warm and cool rather than blink.
        """
        if dial_position not in CHAMBERS:
            raise ValueError(f"dial position {dial_position} is not a chamber")

        for c in CHAMBERS:
            if c == dial_position:
                self._level[c] = min(
                    self.amplitude_mv, self._level[c] + self._step_up
                )
            else:
                self._level[c] = max(0.0, self._level[c] - self._step_up)

        # The operator is never heated. It sits outside.
        self._level[OPERATOR] = 0.0

        self._buf[:] = 0.0
        for c in CHAMBERS:
            if self._level[c] > 0.0:
                self._buf[self.target_indices, c] = self._level[c]
        return self._buf

    @property
    def levels(self) -> np.ndarray:
        """Per-instance heat level. Drives the chamber rendering."""
        return self._level.copy()


@dataclass
class ContinuousHeat:
    """Five chambers, each held at its own level. Nothing is ever off.

    Replaces the single-selection Heat above. The operator sets five levels in
    0..1 and each chamber is heated to its own, continuously, so every chamber
    accumulates rather than waiting its turn.

    Isolation is still structural: this writes chamber columns only and never
    the operator's, so there is no path by which the operator can be heated.
    """

    target_indices: np.ndarray
    n_neurons: int
    n_instances: int
    amplitude_mv: float = 8.0

    def __post_init__(self) -> None:
        if len(self.target_indices) == 0:
            raise ValueError(
                "no thermosensory targets resolved. Check the TRN_VP prefix; "
                "note that type 'AC' does not exist in MaleCNS v1.0."
            )
        self._buf = np.zeros((self.n_neurons, self.n_instances), dtype=np.float32)
        self._levels = np.zeros(self.n_instances, dtype=np.float32)

    def injection(self, levels: np.ndarray) -> np.ndarray:
        """levels: (n_chambers,) in 0..1. Returns (N, n_instances) injection."""
        levels = np.asarray(levels, dtype=np.float32)
        if len(levels) != len(CHAMBERS):
            raise ValueError(
                f"expected {len(CHAMBERS)} levels, got {len(levels)}"
            )

        self._buf[:] = 0.0
        for c in CHAMBERS:
            self._levels[c] = float(np.clip(levels[c], 0.0, 1.0))
            if self._levels[c] > 0.0:
                self._buf[self.target_indices, c] = (
                    self._levels[c] * self.amplitude_mv
                )

        # The operator sits outside. It is never heated at any level.
        self._levels[OPERATOR] = 0.0
        return self._buf

    @property
    def levels(self) -> np.ndarray:
        """Per-instance heat in mV terms, for rendering."""
        return self._levels * self.amplitude_mv


@dataclass
class Electrode:
    """Discrete stimulation pulses from a point electrode.

    Replaces ContinuousHeat, and the change is not cosmetic.

    Heat implied cooking, which this model cannot represent: there is no
    tissue damage, no nociception, and nothing that degrades. A neuron here
    integrates input and spikes and cannot be injured. See
    docs/what-heat-is.md.

    What the simulation literally does is inject millivolts of depolarising
    current. That was always the mechanism; heat was an interpretive layer on
    top of it. Millivolts is the real unit, so an electrode is the more honest
    description rather than the more lurid one.

    Two things make this defensible where the heat channel was not:

    **It has a position.** Every neuron is stimulated in proportion to its
    distance from the electrode tip, on a Gaussian falloff, which is what a
    real electrode does. Nothing is selected by cell type, so there is no "why
    those neurons?" to answer. The thermoreceptors it replaces had no soma
    coordinates at all, 0 of 25, because they are afferents whose cell bodies
    sit in the periphery: the old channel could not have been placed even in
    principle.

    **It is discrete.** Real stimulation is delivered as pulses, not as a
    held level. The chamber level now sets how OFTEN a chamber is pulsed
    rather than how hard, so a chamber at full is being hit constantly and one
    at 20% is hit occasionally. Amplitude per pulse is fixed, which is also
    how a stimulator is used.

    Isolation is unchanged and still structural: this writes chamber columns
    only and never the operator's.
    """

    soma_xyz: np.ndarray          # (N, 3), NaN where unknown
    site: np.ndarray              # (3,) electrode tip
    n_neurons: int
    n_instances: int
    amplitude_mv: float = 8.0
    radius: float = 6000.0
    # Motor neurons driven alongside the electrode, so the body actually moves.
    #
    # This is an authored intervention and the README says so. It is NOT the
    # escape reflex: the escape circuit is present in the dataset and driving
    # it does nothing here. Measured, the 12 escape command neurons give 3,713
    # motor spikes at 20 mV and 3,686 at 400 mV, saturating immediately, and
    # against the phasic baseline the real runs use the result is 0.92x, very
    # slightly FEWER motor spikes than with escape off, because the network is
    # already saturated and the extra input lands on refractory neurons.
    #
    # Driving the 708 VNC motor neurons directly does scale: 1.20x at 10 mV,
    # 1.43x at 25 mV, 1.75x at 60 mV. That is current injected into the output
    # stage, bypassing the command neurons that would normally decide to fire
    # it, which is the honest description and the one the README carries.
    motor_indices: np.ndarray | None = None
    motor_mv: float = 60.0
    # Pulse shape. 0.4ms at dt=0.1ms is 4 steps, within the range used for
    # real intracellular stimulation and long enough to matter against a 20ms
    # membrane time constant.
    pulse_steps: int = 4
    # Pulse rate at full level, in steps between pulse onsets. 200 steps is
    # 20ms, so 50Hz at maximum, falling to nothing at zero.
    min_period: int = 200
    max_period: int = 2000

    def __post_init__(self) -> None:
        xyz = np.asarray(self.soma_xyz, dtype=np.float64)
        if xyz.shape != (self.n_neurons, 3):
            raise ValueError(
                f"soma_xyz must be ({self.n_neurons}, 3), got {xyz.shape}"
            )
        d = np.linalg.norm(xyz - np.asarray(self.site, dtype=np.float64), axis=1)
        # Neurons with no coordinate receive nothing. This is declared rather
        # than hidden: neuPrint has no soma position for 14.9% of the network,
        # so the electrode cannot reach them.
        w = np.exp(-((d / self.radius) ** 2))
        w[~np.isfinite(d)] = 0.0
        self._weights = w.astype(np.float32)
        self.reached = int((self._weights > 0.01).sum())
        if self.reached == 0:
            raise ValueError(
                "electrode reaches no neurons; check the site and radius "
                "against data/soma.npz"
            )
        self._buf = np.zeros((self.n_neurons, self.n_instances), dtype=np.float32)
        self._levels = np.zeros(self.n_instances, dtype=np.float32)
        # Phase per chamber, so the five do not pulse in lockstep. Seeded from
        # the chamber index, not from an RNG: there is no randomness anywhere
        # in this pipeline and adding one here would undermine the claim that
        # divergence is driven by spikes alone.
        self._phase = np.array(
            [int(c * self.min_period / max(len(CHAMBERS), 1)) for c in CHAMBERS],
            dtype=np.int64,
        )
        self._firing = np.zeros(len(CHAMBERS), dtype=np.int64)
        self._motor = (
            None if self.motor_indices is None
            else np.asarray(self.motor_indices, dtype=np.int64)
        )

    def injection(self, levels: np.ndarray, step: int) -> np.ndarray:
        """levels: (n_chambers,) in 0..1. Returns (N, n_instances) injection.

        The level sets pulse RATE, not amplitude. A chamber at full is pulsed
        every min_period steps; one near zero is barely pulsed at all.
        """
        levels = np.asarray(levels, dtype=np.float32)
        if len(levels) != len(CHAMBERS):
            raise ValueError(
                f"expected {len(CHAMBERS)} levels, got {len(levels)}"
            )

        self._buf[:] = 0.0
        for c in CHAMBERS:
            lv = float(np.clip(levels[c], 0.0, 1.0))
            self._levels[c] = lv
            if lv <= 0.0:
                continue
            # Interpolate the period so a higher level fires more often.
            period = int(round(
                self.max_period + (self.min_period - self.max_period) * lv
            ))
            period = max(self.pulse_steps + 1, period)
            if (step + int(self._phase[c])) % period < self.pulse_steps:
                self._buf[:, c] = self._weights * self.amplitude_mv
                self._firing[c] = step

            # The motor stage is driven CONTINUOUSLY, not only on the pulse.
            #
            # Driving it on the pulse alone gave 1.16x and saturated there,
            # because at 50 Hz a pulse is 4 steps in 200 and the drive was off
            # 98% of the time. A body that only moves during a 0.4 ms window
            # is not convulsing, it is flickering.
            #
            # Held between pulses it reaches the 1.75x measured standalone.
            # Scaled by level, so a chamber at 20% twitches and one at full
            # thrashes, and scaled again inside the pulse so the discharge
            # still lands as a kick on top of the sustained drive.
            if self._motor is not None:
                kick = 1.6 if self._firing[c] == step else 1.0
                self._buf[self._motor, c] += self.motor_mv * lv * kick

        self._levels[OPERATOR] = 0.0
        return self._buf

    def pulsing(self, step: int) -> np.ndarray:
        """(n_chambers,) bool: which chambers are mid-pulse this step."""
        return (step - self._firing) < self.pulse_steps

    @property
    def levels(self) -> np.ndarray:
        """Per-instance level in mV terms, for rendering."""
        return self._levels * self.amplitude_mv
