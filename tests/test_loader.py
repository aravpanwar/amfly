"""Gate 4: counts, and Gate 5: biological sanity.

These run against the real 1.1GB connectome, so they are marked `slow` and
skipped when the data is not present. Everything here was measured from the
v1.0 files; a changed or partial download fails immediately rather than
quietly producing a different piece.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from amfly.config import MEASURED, THERMO_PREFIX
from amfly.data.loader import load, verify_counts

DATA_DIR = Path(os.environ.get("AMFLY_DATA", "data"))
_HAVE_DATA = (DATA_DIR / "weights.feather").exists()

pytestmark = pytest.mark.skipif(
    not _HAVE_DATA, reason=f"connectome not found in {DATA_DIR}; set AMFLY_DATA"
)


@pytest.fixture(scope="module")
def connectome():
    return load(DATA_DIR)


def test_gate4_counts_exact(connectome):
    """Not approximate. These are the four numbers the README publishes."""
    verify_counts(connectome)
    assert connectome.n == MEASURED.neurons
    assert connectome.provenance["connections"] == MEASURED.connections
    assert connectome.provenance["synapses"] == MEASURED.synapses


def test_synapse_count_is_exact_not_float_rounded(connectome):
    """Synapses are summed in int64.

    Summing 25.6M integers in fp32 gives 124,177,632, which is 15 too high,
    because fp32 cannot represent every integer above 2**24. Same
    non-associativity the whole project is built around, appearing in the
    first arithmetic the loader does.
    """
    assert connectome.provenance["synapses"] == 124_177_617


def test_descending_neurons(connectome):
    assert len(connectome.descending_indices()) == MEASURED.descending_neurons


def test_thermo_targets_resolve(connectome):
    """AC neurons do not exist here; TRN_VP is the real heat channel."""
    idx = connectome.thermo_indices()
    assert len(idx) == 25
    types = sorted(set(connectome.cell_type[idx]))
    assert types == ["TRN_VP1m", "TRN_VP2", "TRN_VP3a", "TRN_VP3b"]
    assert all(t.startswith(THERMO_PREFIX) for t in types)


def test_no_optic_lobe_contamination(connectome):
    """Guards the substring trap: LoVP* must never enter the heat channel."""
    idx = connectome.thermo_indices()
    assert not any(
        str(t).startswith("LoVP") for t in connectome.cell_type[idx]
    ), "optic lobe visual neurons leaked into the thermosensory targets"


def test_csr_is_canonical(connectome):
    assert connectome.csr.has_sorted_indices
    assert connectome.csr.shape == (MEASURED.neurons, MEASURED.neurons)
    assert connectome.csr.dtype == np.float32


def test_sign_policy_totals(connectome):
    """The unclear/missing default is declared, so assert the count it touches."""
    assert connectome.provenance["unclear_nt_neurons"] == 3_177
    signs = set(np.unique(connectome.sign).tolist())
    assert signs <= {-1.0, 1.0, 0.0}
