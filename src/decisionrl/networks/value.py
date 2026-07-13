"""State-value and continuous action-value (critic) networks."""

from __future__ import annotations

from typing import Sequence, Type

import torch
import torch.nn as nn

from .mlp import build_mlp

__all__ = ["VNetwork", "ContinuousQ"]


class VNetwork(nn.Module):
    """State-value function V(s)."""

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: Type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        self.net = build_mlp(obs_dim, 1, hidden_sizes, activation, output_gain=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class ContinuousQ(nn.Module):
    """Action-value function Q(s, a) for continuous actions."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Sequence[int] = (256, 256),
        activation: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.net = build_mlp(obs_dim + act_dim, 1, hidden_sizes, activation, output_gain=1.0)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1)).squeeze(-1)
