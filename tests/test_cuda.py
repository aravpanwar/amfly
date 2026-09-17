"""The determinism gates, on the GPU.

Skipped when CUDA is unavailable, so CI and CPU-only machines stay green.

A GPU backend has to keep the property the whole piece
rests on. Speed that costs bit-identity is worse than no speed at all, because
the six instances would decorrelate from arithmetic rather than from heat and
the result would look like success.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from amfly.config import LIF
from amfly.sim.engine_cuda import CudaEngine, available

pytestmark = pytest.mark.skipif(not available(), reason="CUDA not available")


@pytest.fixture(scope="module", autouse=True)
def _determinism():
    from amfly.sim.backend import configure_torch_determinism

    configure_torch_determinism()


def _toy(n: int = 1500, seed: int = 0) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    m = sp.random(n, n, density=0.01, format="csr", dtype=np.float32,
                  random_state=rng)
    m.data = (rng.integers(1, 6, size=m.nnz) *
              rng.choice([1.0, -1.0], size=m.nnz)).astype(np.float32)
    m.sort_indices()
    return m


def _run(csr, steps=200, heat_instance=None):
    import torch

    eng = CudaEngine(csr, LIF())
    st = eng.initial_state()
    n = csr.shape[0]
    out = []
    for t in range(steps):
        inj = torch.zeros((n, 6), dtype=torch.float32, device="cuda")
        if t < 40:
            inj[:30, :] = 3.0
        if heat_instance is not None and 50 <= t < 150:
            inj[:15, heat_instance] += 4.0
        out.append(eng.step(st, inj).cpu().numpy())
    return np.stack(out)


def test_gpu_instances_are_bit_identical():
    r = _run(_toy())
    for i in range(1, 6):
        assert np.array_equal(r[:, :, 0], r[:, :, i]), (
            f"instance {i} diverged on GPU under identical input. "
            "Check TF32 is off and deterministic algorithms are on."
        )
    assert r.sum() > 0, "network silent; gate passed vacuously"


def test_gpu_isolation():
    csr = _toy()
    base = _run(csr)
    hot = _run(csr, heat_instance=3)
    for i in (0, 1, 2, 4, 5):
        assert np.array_equal(hot[:, :, i], base[:, :, i])
    assert not np.array_equal(hot[:, :, 3], base[:, :, 3])


def test_gpu_rerun_reproduces_itself():
    csr = _toy()
    assert np.array_equal(_run(csr), _run(csr))


def test_tf32_is_disabled():
    """TF32 truncates fp32 mantissa bits and would break bit-identity."""
    import torch

    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False


def test_gpu_and_cpu_agree_on_spike_counts():
    """Not bit-identity across backends, which is not promised.

    Different hardware reduces in a different order, so the two backends can
    disagree on individual spikes. They should still land in the same regime;
    a large gap means one of them is wrong rather than merely reordered.
    """
    from amfly.sim.engine import Engine, State

    csr = _toy()
    gpu = _run(csr).sum()

    lif = LIF()
    eng = Engine(csr, lif)
    st = State.initial(csr.shape[0], lif)
    total = 0
    for t in range(200):
        inj = np.zeros((csr.shape[0], 6), dtype=np.float32)
        if t < 40:
            inj[:30, :] = 3.0
        total += eng.step(st, inj).sum()

    assert gpu > 0 and total > 0
    ratio = gpu / total
    assert 0.5 < ratio < 2.0, f"backends disagree by {ratio:.2f}x, not reordering"


def test_gpu_matches_cpu_exactly_on_this_hardware():
    """Observed, not promised. See docs/decisions.md.

    Both backends reduce in CSR index order, and on this machine that turns out
    to produce byte-identical results. The design does not depend on it, but a
    regression here would mean one backend's reduction order changed, which is
    worth knowing about.
    """
    from amfly.sim.engine import Engine, State

    csr = _toy()
    gpu = _run(csr, steps=120)

    lif = LIF()
    eng = Engine(csr, lif)
    st = State.initial(csr.shape[0], lif)
    cpu = []
    for t in range(120):
        inj = np.zeros((csr.shape[0], 6), dtype=np.float32)
        if t < 40:
            inj[:30, :] = 3.0
        cpu.append(eng.step(st, inj))
    cpu = np.stack(cpu)

    assert np.array_equal(gpu, cpu), (
        "GPU and CPU diverged. Not a correctness failure by itself, since "
        "cross-backend identity is not promised, but it means a reduction "
        "order changed and the cause should be understood."
    )
