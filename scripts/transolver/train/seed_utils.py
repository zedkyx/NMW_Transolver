"""Global RNG locking for reproducible KETI training."""

from __future__ import annotations

import os
import random
from typing import Callable

import numpy as np
import torch


def set_global_seed(
    seed: int,
    *,
    deterministic: bool = False,
    rank: int = 0,
) -> int:
    """
    Lock python / numpy / torch / cuda RNGs.

    Effective seed = seed + rank (DDP ranks diverge intentionally).
    Returns the effective seed used.

    deterministic=False (default): seed only. Fixes weight init / shuffle.
    deterministic=True: also force slow deterministic CUDA kernels; unsafe on
    a display GPU (watchdog timeout).
    """
    effective = int(seed) + int(rank)

    # Must be set before the first CUDA matmul for full cuBLAS determinism.
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(effective)
    np.random.seed(effective)
    torch.manual_seed(effective)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only: some ops lack a deterministic kernel; fail soft with a warning.
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            # Older torch: no warn_only
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
    else:
        # Leave cuDNN at PyTorch defaults (benchmark=False). Do not enable
        # autotune, and do not switch to slow deterministic kernels.
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass

    return effective


def make_dataloader_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def make_worker_init_fn(base_seed: int) -> Callable[[int], None]:
    """Each DataLoader worker gets a distinct, deterministic seed."""

    def _init(worker_id: int) -> None:
        s = int(base_seed) + int(worker_id)
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)

    return _init
