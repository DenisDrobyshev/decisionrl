"""Ensemble dynamics model for model-based RL (MBPO).

An ensemble of MLPs predicts the next-state delta and reward from ``(state,
action)``. Ensembling captures model uncertainty; rollouts sample a random member
per step so compounding errors stay diverse rather than systematically biased.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn

from .mlp import build_mlp

__all__ = ["EnsembleDynamics"]


class EnsembleDynamics(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        ensemble_size: int = 5,
        hidden_sizes: Sequence[int] = (200, 200),
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.ensemble_size = int(ensemble_size)
        # each model outputs [next-state delta (obs_dim), reward (1)]
        self.models = nn.ModuleList(
            [build_mlp(obs_dim + act_dim, obs_dim + 1, hidden_sizes, activation=nn.ReLU)
             for _ in range(self.ensemble_size)]
        )

    def forward_all(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predictions from every member: shape ``(ensemble, batch, obs_dim + 1)``."""
        x = torch.cat([obs, action], dim=-1)
        return torch.stack([m(x) for m in self.models], dim=0)

    @torch.no_grad()
    def predict(self, obs: torch.Tensor, action: torch.Tensor, member: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Next-state and reward from a single ensemble member (random model rollouts)."""
        out = self.models[member](torch.cat([obs, action], dim=-1))
        delta, reward = out[..., : self.obs_dim], out[..., self.obs_dim]
        return obs + delta, reward
