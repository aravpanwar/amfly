"""MaleCNS v1.0 -> filtered, signed, canonically ordered CSR.

This is the ONLY dataset-aware module. Moving to BANC or FlyWire later should
touch this file and nothing else.

Two traps live here, both found by measurement rather than by reading docs:

1. The weights table is 151,856,684 rows, not 25.6M. Most rows involve segments
   that are not traced neurons. Loading it naively as int64 costs ~3.6GB. We
   project to three columns, filter both endpoints, then downcast.

2. Row order is load-bearing. We sort bodyId ascending before assigning indices,
   which makes the CSR ordering canonical. Without it, the "bit-identical across
   machines" claim is false and the whole piece is unverifiable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import scipy.sparse as sp

from ..config import (
    EXCITATORY,
    INHIBITORY,
    MEASURED,
    SUPERCLASS_DESCENDING,
    THERMO_PREFIX,
    HYGRO_PREFIX,
    UNCLEAR_SIGN,
)

log = logging.getLogger(__name__)


@dataclass
class Connectome:
    """One shared graph. Six instances will share this exact object."""

    csr: sp.csr_matrix  # (N, N) float32, signed, weights in synapse counts
    body_ids: np.ndarray  # (N,) int64, sorted ascending. index i <-> body_ids[i]
    superclass: np.ndarray  # (N,) object
    cell_type: np.ndarray  # (N,) object
    sign: np.ndarray  # (N,) float32, +1/-1 per PREsynaptic neuron
    provenance: dict

    @property
    def n(self) -> int:
        return len(self.body_ids)

    def index_of(self, body_id: int) -> int:
        """Body id -> row index. Valid because body_ids is sorted."""
        i = int(np.searchsorted(self.body_ids, body_id))
        if i >= len(self.body_ids) or self.body_ids[i] != body_id:
            raise KeyError(f"body {body_id} is not a retained neuron")
        return i

    def descending_indices(self) -> np.ndarray:
        return np.flatnonzero(self.superclass == SUPERCLASS_DESCENDING)

    def by_type_prefix(self, prefix: str) -> np.ndarray:
        """Indices whose cell type starts with `prefix`.

        startswith, never 'in'. Matching "VP" as a substring would pull in the
        LoVP* optic-lobe family and quietly turn the heat channel into a light
        channel.
        """
        t = self.cell_type
        keep = np.array(
            [isinstance(x, str) and x.startswith(prefix) for x in t], dtype=bool
        )
        return np.flatnonzero(keep)

    def thermo_indices(self, include_hygro: bool = False) -> np.ndarray:
        idx = self.by_type_prefix(THERMO_PREFIX)
        if include_hygro:
            idx = np.union1d(idx, self.by_type_prefix(HYGRO_PREFIX))
        return idx


def _nt_sign(nt: object) -> float:
    if isinstance(nt, str):
        if nt in EXCITATORY:
            return 1.0
        if nt in INHIBITORY:
            return -1.0
    # "unclear", or no prediction at all. Declared default, counted by caller.
    return UNCLEAR_SIGN


def load(
    data_dir: Path,
    weight_threshold: int = 1,
    silence_unclear_nt: bool = False,
) -> Connectome:
    """Build the shared connectome. Expect ~2 minutes cold, mostly the join."""
    data_dir = Path(data_dir)

    # --- neurons -----------------------------------------------------------
    ann = feather.read_table(
        data_dir / "ann.feather",
        columns=["bodyId", "superclass", "type", "class", "somaSide"],
    ).to_pandas()
    log.info("annotations: %d raw rows", len(ann))

    neurons = ann[ann["superclass"].notna()].copy()
    # THE canonical ordering. Everything downstream depends on this sort.
    neurons = neurons.sort_values("bodyId", kind="mergesort").reset_index(drop=True)
    body_ids = neurons["bodyId"].to_numpy(dtype=np.int64)
    n = len(body_ids)
    log.info("retained neurons: %d", n)

    # --- neurotransmitter sign, per presynaptic neuron (Dale's law) ---------
    nt = feather.read_table(
        data_dir / "nt.feather", columns=["body", "consensus_nt"]
    ).to_pandas()
    nt = nt.drop_duplicates("body")

    merged = neurons[["bodyId"]].merge(
        nt, left_on="bodyId", right_on="body", how="left"
    )
    nt_values = merged["consensus_nt"].to_numpy()
    sign = np.array([_nt_sign(x) for x in nt_values], dtype=np.float32)

    unclear_mask = np.array(
        [not (isinstance(x, str) and (x in EXCITATORY or x in INHIBITORY))
         for x in nt_values],
        dtype=bool,
    )
    n_unclear = int(unclear_mask.sum())
    log.info(
        "unclear/missing neurotransmitter: %d neurons (%.2f%%), default sign %+g",
        n_unclear,
        100.0 * n_unclear / n,
        UNCLEAR_SIGN,
    )
    if silence_unclear_nt:
        # Sensitivity run: zero them rather than guess a sign.
        sign[unclear_mask] = 0.0
        log.info("sensitivity run: silenced %d unclear neurons", n_unclear)

    # --- edges -------------------------------------------------------------
    # Project to 3 columns before anything else. This is the difference between
    # ~1.2GB and ~3.6GB of resident memory.
    w = feather.read_table(
        data_dir / "weights.feather", columns=["body_pre", "body_post", "weight"]
    ).to_pandas()
    log.info("weights: %d raw rows", len(w))

    order = np.argsort(body_ids, kind="mergesort")  # identity, body_ids is sorted
    assert np.array_equal(order, np.arange(n)), "body_ids must be sorted ascending"

    pre = np.searchsorted(body_ids, w["body_pre"].to_numpy(dtype=np.int64))
    post = np.searchsorted(body_ids, w["body_post"].to_numpy(dtype=np.int64))
    np.clip(pre, 0, n - 1, out=pre)
    np.clip(post, 0, n - 1, out=post)

    valid = (
        (body_ids[pre] == w["body_pre"].to_numpy(dtype=np.int64))
        & (body_ids[post] == w["body_post"].to_numpy(dtype=np.int64))
    )
    weight = w["weight"].to_numpy()
    if weight_threshold > 1:
        valid &= weight >= weight_threshold
        log.info("weight threshold %d applied (declare this)", weight_threshold)

    pre = pre[valid].astype(np.int32, copy=False)
    post = post[valid].astype(np.int32, copy=False)

    # Count synapses in int64 BEFORE casting to float32. Summing 25.6M integers
    # in fp32 loses precision (measured: 124,177,632 vs the true 124,177,617,
    # off by 15) because fp32 cannot represent every integer above 2**24. That
    # is the same non-associativity this whole project is built around, showing
    # up in the very first arithmetic we do.
    weight_i = weight[valid].astype(np.int64, copy=False)
    n_synapses = int(weight_i.sum())

    weight = weight_i.astype(np.float32, copy=False)
    log.info("edges after filter: %d", len(weight))
    log.info("synapses after filter: %d", n_synapses)

    del w

    # Signed by the PREsynaptic neuron's transmitter.
    data = weight * sign[pre]

    # rows = postsynaptic, so csr @ state gathers each neuron's own inputs in a
    # fixed index order. That fixed order is the determinism guarantee.
    csr = sp.csr_matrix((data, (post, pre)), shape=(n, n), dtype=np.float32)
    csr.sort_indices()

    provenance = {
        "neurons": int(n),
        "connections": int(len(weight)),
        "synapses": n_synapses,
        "unclear_nt_neurons": n_unclear,
        "unclear_nt_sign": UNCLEAR_SIGN if not silence_unclear_nt else 0.0,
        "weight_threshold": weight_threshold,
        "nnz": int(csr.nnz),
    }

    return Connectome(
        csr=csr,
        body_ids=body_ids,
        superclass=neurons["superclass"].to_numpy(),
        cell_type=neurons["type"].to_numpy(),
        sign=sign,
        provenance=provenance,
    )


def verify_counts(c: Connectome) -> None:
    """Gate 4. A changed or partial download fails here, immediately."""
    assert c.n == MEASURED.neurons, f"neurons {c.n} != {MEASURED.neurons}"
    assert (
        c.provenance["connections"] == MEASURED.connections
    ), f"connections {c.provenance['connections']} != {MEASURED.connections}"
    assert (
        c.provenance["synapses"] == MEASURED.synapses
    ), f"synapses {c.provenance['synapses']} != {MEASURED.synapses}"
    n_dn = len(c.descending_indices())
    assert (
        n_dn == MEASURED.descending_neurons
    ), f"DNs {n_dn} != {MEASURED.descending_neurons}"
