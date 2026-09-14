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
