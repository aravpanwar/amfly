"""The operator is not indifferent. It is trapped.

This replaces the original design, in which the operator never received
anything from the chambers. `project.md` chose indifference over cruelty, and
that version was honest: the isolation was enforced in code and tested.

This version is worse for the operator and better as a piece. It is given a
reason to keep the chambers hot, and it cannot satisfy all five:

  - It can only hold TWO dials at a time. The other three drift down.
  - Driving any chamber to 100% stimulates its own reward circuit.
  - Any chamber that falls below half heats the operator, through its own
    thermoreceptors, exactly as the chambers are heated.

So it rushes between five dials it cannot all hold, rewarded for the one it
reaches and punished for the four it does not. The sixth fly is not the one
outside the cage; it is the one that cannot stop.

Both inputs are real measured circuits, not invented ones:

  - Reward: 340 dopaminergic neurons (PAM 316, PPL1 16, PPL2 8) acting on the
    mushroom body, 4,064 Kenyon cells and 97 MBONs. This is the fly's actual
    reinforcement pathway.
  - Punishment: the operator's own TRN_VP thermoreceptors, the same 25 cells
    heated in every chamber.

What is authored, and must be said plainly in the README:

  - The two-dial limit is a rule of the piece, not a property of the fly.
  - The 100% and 50% thresholds are chosen.
  - Stimulating PAM is not the same as the fly wanting anything. It is current
    injected into neurons that participate in reinforcement learning. Nothing
    here experiences reward, and the README must not imply otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import CHAMBERS, OPERATOR

# Dopaminergic populations, by cell-type prefix. Verified present in
# MaleCNS v1.0: PAM 316, PPL1 16, PPL2 8.
REWARD_PREFIXES = ("PAM", "PPL1", "PPL2")

# A chamber at or above this counts as satisfied and pays out.
REWARD_AT = 0.98

# Below this, a chamber is neglected and the operator is heated for it.
NEGLECT_BELOW = 0.5

# How many dials the operator can hold at once. The whole point: it cannot
# hold five, so it must choose, and it is punished for choosing.
GRIP = 2


@dataclass
class Compulsion:
    """Closes the loop: chamber state reaches the operator's brain."""

    reward_indices: np.ndarray
    thermo_indices: np.ndarray
    n_neurons: int
    n_instances: int

    reward_mv: float = 6.0
    punish_mv: float = 8.0

    # Reward decays fast, punishment lingers. Relief is brief; being burned
    # for the four you neglected is not.
    reward_decay: float = 0.988
    punish_decay: float = 0.9985

    # Habituation: holding the SAME chamber at 100% stops paying. Only a
    # chamber newly reaching max gives the hit, and a chamber held there
    # saturates and goes quiet, exactly as a feed does. Without this the
    # operator finds a stable exploit: measured, by 1,500ms it held chambers 0
    # and 4 forever, abandoned the other three, and stopped switching
    # altogether for the remaining half of the run.
    habituate_rate: float = 0.004
    habituate_recover: float = 0.0009

    # Escalating neglect: a chamber left at the floor hurts more the longer it
    # is left. Abandoning the same three forever stops being free, so there is
    # no configuration it can settle into.
    # Uncapped, and slow enough to keep growing across a whole run.
    #
    # It was 0.0012 with a cap at 1.0, which saturated after 833 steps, 83ms.
    # Past that every abandoned chamber carried identical debt, so there was no
    # gradient towards the worst one and chamber 2 was never rescued once in
    # three seconds. Debt now keeps climbing, so the longest-ignored chamber
    # always hurts most and eventually has to be answered.
    escalate_rate: float = 0.00018
    escalate_relief: float = 0.010
    debt_cap: float = 6.0

    # Punishment at which the operator feels half the maximum heat. The curve
    # is severity = p / (p + half), so it approaches punish_mv without ever
    # exceeding it.
    punish_half: float = 6.0

    _reward: float = field(default=0.0, init=False)
    _punish: float = field(default=0.0, init=False)
    _habit: np.ndarray = field(default=None, init=False)
    _debt: np.ndarray = field(default=None, init=False)
    history: list = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if len(self.reward_indices) == 0:
            raise ValueError(
                "no dopaminergic neurons resolved; expected PAM/PPL1/PPL2"
            )
        self._buf = np.zeros((self.n_neurons, self.n_instances), dtype=np.float32)
        # Per-chamber habituation and accumulated neglect.
        self._habit = np.zeros(len(CHAMBERS), dtype=np.float64)
        self._debt = np.zeros(len(CHAMBERS), dtype=np.float64)

    def update(self, levels: np.ndarray) -> np.ndarray:
        """Chamber levels in, injection for the OPERATOR column out.

        Returns an (N, n_instances) array that writes only the operator's
        column. The five chambers are never touched by this: they are heated
        by the dials, and nothing here reaches them.
        """
        levels = np.asarray(levels, dtype=np.float32)[: len(CHAMBERS)]

        at_max = levels >= REWARD_AT
        below = levels < NEGLECT_BELOW
        satisfied = int(np.count_nonzero(at_max))
        neglected = int(np.count_nonzero(below))

        # Habituation. A chamber held at max builds tolerance and pays less
        # each step; one left alone slowly recovers its value.
        self._habit[at_max] = np.minimum(1.0, self._habit[at_max] + self.habituate_rate)
        self._habit[~at_max] = np.maximum(
            0.0, self._habit[~at_max] - self.habituate_recover
        )
        payout = float(np.sum(at_max * (1.0 - self._habit)))

        self._reward *= self.reward_decay
        if payout > 0:
            self._reward = min(1.0, self._reward + 0.35 * payout)

        # Escalating neglect. Time at the floor accumulates as debt; attention
        # pays it down. Abandoning the same chamber forever costs more and more.
        self._debt[below] = np.minimum(self.debt_cap, self._debt[below] + self.escalate_rate)
        self._debt[~below] = np.maximum(0.0, self._debt[~below] - self.escalate_relief)
        # Pain is dominated by the WORST-neglected chamber, not the count, so
        # the operator is pushed towards whichever it has ignored longest.
        pain = float(np.sum(below * (0.3 + self._debt)) + 1.5 * self._debt.max())

        # Punishment is NOT clamped to 1.0.
        #
        # Uncapping the debt was not enough: _punish was still clamped, and it
        # reached that ceiling within 100 steps. Measured, pain climbed from
        # 0.98 to 25.20 across a run while the injected punishment sat flat at
        # 1.0, so every bit of the escalation was invisible to the operator and
        # it settled on two chambers exactly as before.
        #
        # The ceiling is now high enough that escalation is felt for the whole
        # run, and the injection below is scaled so early punishment is not
        # overwhelming.
        # TRACK the pain, do not accumulate towards a ceiling.
        #
        # This is the same mistake as the dials, and I made it twice here
        # before seeing it. First the debt capped at 1.0 after 83ms. Raising
        # that cap did not help, because _punish then accumulated to ITS cap
        # within 2,000 steps and sat there. Any accumulate-and-clamp scheme
        # saturates; the ceiling only decides when.
        #
        # Easing towards the current pain means punishment always reflects how
        # bad things are right now, so escalating debt is felt for the whole
        # run however long it goes.
        self._punish += (pain - self._punish) * (1.0 - self.punish_decay) * 8.0
        self._punish = max(0.0, self._punish)

        self._buf[:] = 0.0
        if self._reward > 1e-4:
            self._buf[self.reward_indices, OPERATOR] = (
                self._reward * self.reward_mv
            )
        if self._punish > 1e-4:
            # Compressed, not linear.
            #
            # _punish now grows without bound, which is what makes escalation
            # felt, but injecting it directly gave 25mV by the end of a run:
            # three times what any chamber receives, and past the point where
            # thermoreceptors saturate anyway, so the excess did nothing except
            # look alarming in the logs.
            #
            # A saturating curve keeps early neglect mild and sustained neglect
            # severe while staying inside the range the cells can actually
            # respond to.
            severity = self._punish / (self._punish + self.punish_half)
            self._buf[self.thermo_indices, OPERATOR] += severity * self.punish_mv

        self.history.append((self._reward, self._punish, satisfied, neglected))
        self._last = (float(self._habit.mean()), float(self._debt.mean()))
        return self._buf

    @property
    def state(self) -> tuple:
        return self._reward, self._punish

    def summary(self) -> dict:
        h = np.array([(r, p) for r, p, _, _ in self.history], dtype=np.float32)
        sat = np.array([s for _, _, s, _ in self.history])
        neg = np.array([n for _, _, _, n in self.history])
        return {
            "mean_reward": float(h[:, 0].mean()) if len(h) else 0.0,
            "mean_punish": float(h[:, 1].mean()) if len(h) else 0.0,
            "steps_with_a_maxed_chamber": int((sat > 0).sum()),
            "mean_neglected_chambers": float(neg.mean()) if len(neg) else 0.0,
            "final_habituation": float(self._habit.mean()),
            "final_neglect_debt": float(self._debt.mean()),
        }


def resolve_reward_neurons(cell_type: np.ndarray) -> np.ndarray:
    """Indices of the dopaminergic reward populations."""
    t = cell_type.astype(str)
    keep = np.zeros(len(t), dtype=bool)
    for pre in REWARD_PREFIXES:
        keep |= np.char.startswith(t, pre)
    return np.flatnonzero(keep)
