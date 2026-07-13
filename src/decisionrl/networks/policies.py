"""Policy network heads: categorical, Gaussian, squashed-Gaussian, deterministic."""

from __future__ import annotations

from typing import Sequence, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from .mlp import build_mlp, layer_init

__all__ = [
    "CategoricalActor",
    "GaussianActor",
    "SquashedGaussianActor",
    "DeterministicActor",
]


class CategoricalActor(nn.Module):
    """Stochastic policy over a discrete action set."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: Type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        # Small output gain (0.01) keeps the initial policy close to uniform.
        self.net = build_mlp(obs_dim, n_actions, hidden_sizes, activation, output_gain=0.01)

    def distribution(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self.net(obs))

    def forward(self, obs: torch.Tensor) -> Categorical:
        return self.distribution(obs)


class GaussianActor(nn.Module):
    """Diagonal-Gaussian policy with a state-independent log-std.

    Used by on-policy continuous-control algorithms (A2C/PPO). Actions are not
    squashed; the caller is responsible for clipping to the action bounds when
    stepping the environment.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: Type[nn.Module] = nn.Tanh,
        log_std_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.mean_net = build_mlp(obs_dim, act_dim, hidden_sizes, activation, output_gain=0.01)
        self.log_std = nn.Parameter(torch.ones(act_dim) * float(log_std_init))

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.mean_net(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)

    def forward(self, obs: torch.Tensor) -> Normal:
        return self.distribution(obs)


class SquashedGaussianActor(nn.Module):
    """Tanh-squashed Gaussian policy for SAC (bounded, reparameterized).

    Produces a state-dependent mean and log-std, samples with the
    reparameterization trick, squashes through ``tanh`` and applies the exact
    change-of-variables correction to the log-probability.
    """

    LOG_STD_MIN = -20.0
    LOG_STD_MAX = 2.0

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        hidden_sizes: Sequence[int] = (256, 256),
        activation: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        hidden = list(hidden_sizes)
        self.trunk = build_mlp(obs_dim, hidden[-1], hidden[:-1], activation, output_gain=np.sqrt(2))
        self.act = activation()
        self.mean_head = layer_init(nn.Linear(hidden[-1], act_dim), gain=0.01)
        self.log_std_head = layer_init(nn.Linear(hidden[-1], act_dim), gain=0.01)

        action_scale = (np.asarray(action_high) - np.asarray(action_low)) / 2.0
        action_bias = (np.asarray(action_high) + np.asarray(action_low)) / 2.0
        self.register_buffer("action_scale", torch.as_tensor(action_scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.as_tensor(action_bias, dtype=torch.float32))

    def _mean_logstd(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.act(self.trunk(obs))
        mean = self.mean_head(z)
        log_std = self.log_std_head(z)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(action, log_prob, deterministic_action)``, all scaled."""
        mean, log_std = self._mean_logstd(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        x = normal.rsample()
        y = torch.tanh(x)
        action = y * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x)
        # tanh change-of-variables correction
        log_prob = log_prob - torch.log(self.action_scale * (1 - y.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        deterministic = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, deterministic

    def forward(self, obs: torch.Tensor):
        return self.sample(obs)


class DeterministicActor(nn.Module):
    """Deterministic policy mapping observations to bounded actions (DDPG/TD3)."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        hidden_sizes: Sequence[int] = (256, 256),
        activation: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.net = build_mlp(obs_dim, act_dim, hidden_sizes, activation, output_gain=0.01)
        action_scale = (np.asarray(action_high) - np.asarray(action_low)) / 2.0
        action_bias = (np.asarray(action_high) + np.asarray(action_low)) / 2.0
        self.register_buffer("action_scale", torch.as_tensor(action_scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.as_tensor(action_bias, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs)) * self.action_scale + self.action_bias
