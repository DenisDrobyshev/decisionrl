"""MLP construction with sensible RL initialization."""

from __future__ import annotations

from typing import Sequence, Type

import numpy as np
import torch.nn as nn

__all__ = ["build_mlp", "layer_init"]


def layer_init(layer: nn.Linear, gain: float = np.sqrt(2), bias: float = 0.0) -> nn.Linear:
    """Orthogonal weight init + constant bias.

    Orthogonal initialization with ``gain=sqrt(2)`` for hidden layers (and small
    gains such as 0.01 on policy outputs) is a well-established best practice for
    on-policy algorithms and improves stability noticeably.
    """
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, bias)
    return layer


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_sizes: Sequence[int] = (64, 64),
    activation: Type[nn.Module] = nn.Tanh,
    output_gain: float = 1.0,
    hidden_gain: float = np.sqrt(2),
    layer_norm: bool = False,
) -> nn.Sequential:
    """Build a fully-connected network with orthogonal initialization."""
    layers: list = []
    last = int(input_dim)
    for h in hidden_sizes:
        layers.append(layer_init(nn.Linear(last, h), gain=hidden_gain))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(activation())
        last = h
    layers.append(layer_init(nn.Linear(last, output_dim), gain=output_gain))
    return nn.Sequential(*layers)
