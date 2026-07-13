"""Global seeding for reproducible experiments."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

__all__ = ["set_seed"]


def set_seed(seed: Optional[int], deterministic: bool = False) -> Optional[int]:
    """Seed Python, NumPy and PyTorch RNGs.

    Parameters
    ----------
    seed:
        The seed. If ``None`` nothing is done and ``None`` is returned.
    deterministic:
        If ``True``, also request deterministic CuDNN/torch algorithms. This can
        slow training down but makes runs bit-for-bit reproducible on the same
        hardware.
    """
    if seed is None:
        return None

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:  # pragma: no cover - older torch
                torch.use_deterministic_algorithms(True)
    except ImportError:  # pragma: no cover - torch always present in practice
        pass

    return seed
