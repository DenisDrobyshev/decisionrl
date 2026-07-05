"""PyTorch helper functions used across algorithms."""

from __future__ import annotations

from typing import Union

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "get_device",
    "to_tensor",
    "soft_update",
    "hard_update",
    "explained_variance",
    "polyak_update",
]


def get_device(device: Union[str, torch.device] = "auto") -> torch.device:
    """Resolve ``"auto"`` to CUDA when available, otherwise CPU."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def to_tensor(x, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert numpy/array-likes to a tensor on ``device`` (no-op-ish for tensors)."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)


@torch.no_grad()
def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    """Polyak averaging: ``target = tau * source + (1 - tau) * target``."""
    for src_p, tgt_p in zip(source.parameters(), target.parameters()):
        tgt_p.mul_(1.0 - tau)
        tgt_p.add_(tau * src_p.data)


# Alias used by the DDPG/TD3/SAC literature.
polyak_update = soft_update


@torch.no_grad()
def hard_update(source: nn.Module, target: nn.Module) -> None:
    """Copy parameters from ``source`` to ``target``."""
    target.load_state_dict(source.state_dict())


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Fraction of the return variance explained by the value predictions.

    Returns a value in (-inf, 1]. 1 is a perfect fit, 0 means no better than
    predicting the mean, negative means worse than the mean. A standard PPO/A2C
    diagnostic.
    """
    y_pred = np.asarray(y_pred).ravel()
    y_true = np.asarray(y_true).ravel()
    var_y = np.var(y_true)
    if var_y == 0:
        return float("nan")
    return float(1.0 - np.var(y_true - y_pred) / var_y)
