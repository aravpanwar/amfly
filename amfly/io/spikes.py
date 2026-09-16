"""Spike recording.

Dump size is rate-dependent and was the reason to measure before choosing a
format. At 166,700 neurons x 6 instances over 30s at dt=0.1ms, a dense bitmask
would be about 37GB, which is out. Sparse (step, neuron) pairs at a measured
~2,000 spikes/step land near 1.4GB, which is workable but still too large to
commit.

So we record two things:

  - The summary: per-step population rate per instance. Small, always on, and
    it is what the divergence plots actually consume.
  - A pinned subset at full resolution: the descending neurons, the
    thermoreceptors, and a fixed deterministic sample. Enough to draw a raster
    without storing the whole network.

The sample is chosen by strided selection rather than an RNG, so there is no
random seed anywhere in the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Recorder:
    """Records the summary plus a pinned subset."""

    n_instances: int
    subset: np.ndarray  # neuron indices recorded at full resolution
    rates: list = field(default_factory=list, init=False)
    heat_levels: list = field(default_factory=list, init=False)
    dial: list = field(default_factory=list, init=False)
    hamming: list = field(default_factory=list, init=False)
    # What the operator is being paid and charged, per step. This is the
    # premise of the piece and it used to survive only as a mean in
    # provenance, so nothing downstream could show it.
    drive: list = field(default_factory=list, init=False)
    # Which buttons are down, as a bitmask. One byte per step.
    held: list = field(default_factory=list, init=False)
    _subset_events: list = field(default_factory=list, init=False)

    @classmethod
    def build(
        cls,
        n_neurons: int,
        n_instances: int,
        dn_indices: np.ndarray,
        thermo_indices: np.ndarray,
        sample: int = 2000,
    ) -> "Recorder":
        stride = max(1, n_neurons // sample)
        strided = np.arange(0, n_neurons, stride, dtype=np.int64)
        subset = np.union1d(
            np.union1d(np.asarray(dn_indices), np.asarray(thermo_indices)), strided
        )
        return cls(n_instances=n_instances, subset=subset)

    def record(
        self,
        step: int,
        spikes: np.ndarray,
        heat: np.ndarray,
        dial_position: int,
        hamming: np.ndarray | None = None,
        reward: float | None = None,
        punish: float | None = None,
        held: np.ndarray | None = None,
    ) -> None:
        self.rates.append(spikes.sum(axis=0).astype(np.int32))
        if hamming is not None:
            self.hamming.append(np.asarray(hamming, dtype=np.int32))
        self.heat_levels.append(heat.astype(np.float32))
        self.dial.append(dial_position)
        if reward is not None:
            self.drive.append((np.float32(reward), np.float32(punish or 0.0)))
        if held is not None:
            # Chamber i is bit i. Written explicitly rather than with
            # packbits, which pads to a full byte and put the five chambers in
            # the HIGH bits: a mask of 24 then decoded as chambers 3 and 4
            # when the operator was in fact holding 0 and 1, silently
            # mirroring the console and misnaming every press.
            m = 0
            for i, on in enumerate(np.asarray(held, dtype=bool)):
                if on:
                    m |= 1 << i
            self.held.append(m)

        sub = spikes[self.subset]
        rows, cols = np.nonzero(sub)
        if len(rows):
            self._subset_events.append(
                np.stack(
                    [
                        np.full(len(rows), step, dtype=np.int32),
                        self.subset[rows].astype(np.int32),
                        cols.astype(np.int8),
                    ],
                    axis=0,
                )
            )

    def save(self, out_dir: Path, provenance: dict | None = None) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if self._subset_events:
            ev = np.concatenate(self._subset_events, axis=1)
        else:
            ev = np.zeros((3, 0), dtype=np.int32)

        path = out_dir / "run.npz"
        np.savez_compressed(
            path,
            rates=np.array(self.rates, dtype=np.int32),
            heat=np.array(self.heat_levels, dtype=np.float32),
            dial=np.array(self.dial, dtype=np.int8),
            hamming=np.array(self.hamming, dtype=np.int32)
            if self.hamming
            else np.zeros((0, self.n_instances), dtype=np.int32),
            drive=np.array(self.drive, dtype=np.float32)
            if self.drive
            else np.zeros((0, 2), dtype=np.float32),
            held=np.array(self.held, dtype=np.uint8)
            if self.held
            else np.zeros(0, dtype=np.uint8),
            subset=self.subset.astype(np.int32),
            events_step=ev[0],
            events_neuron=ev[1],
            events_instance=ev[2],
            **{f"prov_{k}": v for k, v in (provenance or {}).items()},
        )
        return path


def load_run(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}
