"""Backend selection, and the determinism settings that make GPU safe here.

CPU is the reference and is always correct. GPU is roughly 20x faster at this
size, which is the difference between 25 minutes and about 80 seconds for 700ms
of simulated time.

The GPU path is only safe with determinism forced on. Left at defaults, PyTorch
may pick non-deterministic reductions and TF32 silently truncates fp32 mantissa
bits, either of which would decorrelate the six instances from arithmetic noise
rather than from heat. That failure looks exactly like success, so the settings
below are mandatory rather than advisory.

Install note: a CUDA build is required. `torch` from PyPI defaults to CPU-only,
and `torch.cuda.is_available()` returning False means the slow path.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def describe() -> dict:
    """What is actually available, for the run manifest.

    Bit-identity is guaranteed within one machine and configuration. It is not
    promised across different GPUs or torch versions, so the versions are
    recorded rather than assumed.
    """
    info = {"backend": "numpy", "deterministic": True}
    try:
        import torch
    except ImportError:
        info["torch"] = None
        return info

    info["torch"] = torch.__version__
    info["cuda_available"] = bool(torch.cuda.is_available())
    if info["cuda_available"]:
        info["device"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
    return info


def configure_torch_determinism() -> bool:
    """Force deterministic reductions and disable TF32. Call before any work.

    Returns True if CUDA is usable, False if this will run on CPU.
    """
    try:
        import torch
    except ImportError:
        log.info("torch not installed; using the numpy reference backend")
        return False

    torch.use_deterministic_algorithms(True)
    # TF32 truncates fp32 mantissa bits on Ampere and later. Fast, and fatal
    # to a bit-identity claim.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if not torch.cuda.is_available():
        log.warning(
            "torch is CPU-only (%s). This runs about 20x slower. "
            "Install a CUDA build to produce clips at a reasonable rate.",
            torch.__version__,
        )
        return False

    log.info("cuda: %s (torch %s, cuda %s)",
             torch.cuda.get_device_name(0), torch.__version__, torch.version.cuda)
    return True
