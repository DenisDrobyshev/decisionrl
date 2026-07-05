"""Q-value networks for discrete action spaces (DQN family)."""

from __future__ import annotations

from typing import Sequence, Type

import torch
import torch.nn as nn

from .mlp import build_mlp, layer_init

__all__ = ["QNetwork", "DuelingQNetwork", "CategoricalQNetwork"]


class QNetwork(nn.Module):
    """Maps an observation to one Q-value per discrete action."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: Sequence[int] = (128, 128),
        activation: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.net = build_mlp(
            obs_dim, n_actions, hidden_sizes, activation=activation, output_gain=1.0
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class DuelingQNetwork(nn.Module):
    """Dueling architecture: separate value and advantage streams.

    ``Q(s, a) = V(s) + (A(s, a) - mean_a A(s, a))``. The mean-subtraction makes
    the decomposition identifiable and improves learning when many actions have
    similar values (Wang et al., 2016).
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: Sequence[int] = (128, 128),
        activation: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        *feature_sizes, last_hidden = hidden_sizes
        self.features = build_mlp(
            obs_dim, last_hidden, feature_sizes, activation=activation, output_gain=1.0
        )
        self.act = activation()
        self.value = layer_init(nn.Linear(last_hidden, 1), gain=1.0)
        self.advantage = layer_init(nn.Linear(last_hidden, n_actions), gain=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        z = self.act(self.features(obs))
        value = self.value(z)
        advantage = self.advantage(z)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class CategoricalQNetwork(nn.Module):
    """Distributional value network (C51): a categorical distribution over a
    fixed set of return atoms for each action. Returns *logits* of shape
    ``(batch, n_actions, n_atoms)``.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        n_atoms: int = 51,
        hidden_sizes: Sequence[int] = (128, 128),
        activation: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.n_actions = int(n_actions)
        self.n_atoms = int(n_atoms)
        self.net = build_mlp(obs_dim, self.n_actions * self.n_atoms, hidden_sizes, activation=activation)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).view(-1, self.n_actions, self.n_atoms)
