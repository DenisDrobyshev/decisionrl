"""Noisy linear layers and the dueling-distributional Rainbow network.

NoisyLinear (Fortunato et al., 2018) adds learnable factorized-Gaussian noise to
a linear layer for parameter-space exploration (replacing epsilon-greedy). In
eval mode it falls back to the mean weights (deterministic). RainbowNetwork
combines the dueling and distributional (C51) architectures with noisy heads.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["NoisyLinear", "RainbowNetwork"]


class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, sigma0: float = 0.5) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("weight_eps", torch.zeros(out_features, in_features))
        self.register_buffer("bias_eps", torch.zeros(out_features))
        self._reset_parameters(sigma0)
        self.reset_noise()

    def _reset_parameters(self, sigma0: float) -> None:
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(sigma0 / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(sigma0 / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self) -> None:
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_eps.copy_(eps_out.outer(eps_in))
        self.bias_eps.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_eps
            bias = self.bias_mu + self.bias_sigma * self.bias_eps
        else:
            weight, bias = self.weight_mu, self.bias_mu
        return F.linear(x, weight, bias)


class RainbowNetwork(nn.Module):
    """Dueling + distributional value network with noisy heads.

    Returns per-action atom *logits* of shape ``(batch, n_actions, n_atoms)``
    (softmax over the atom dimension gives the return distribution).
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        n_atoms: int = 51,
        hidden_sizes: Sequence[int] = (128, 128),
        sigma0: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_actions = int(n_actions)
        self.n_atoms = int(n_atoms)
        layers: list = []
        last = int(obs_dim)
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        self.feature = nn.Sequential(*layers)
        self.value = NoisyLinear(last, self.n_atoms, sigma0)
        self.advantage = NoisyLinear(last, self.n_actions * self.n_atoms, sigma0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        f = self.feature(obs)
        value = self.value(f).view(-1, 1, self.n_atoms)
        advantage = self.advantage(f).view(-1, self.n_actions, self.n_atoms)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    def reset_noise(self) -> None:
        self.value.reset_noise()
        self.advantage.reset_noise()
