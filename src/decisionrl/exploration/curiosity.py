"""Intrinsic-motivation exploration: Random Network Distillation and ICM.

Both turn *"how surprising is this transition?"* into an **intrinsic reward** that
supplements a (possibly sparse) extrinsic reward, driving the agent toward novel
states. They are deliberately agent-agnostic: wrap any environment with
:class:`CuriosityWrapper` and the bonus is folded into the reward stream, so
*every* algorithm in the library (DQN, PPO, SAC, ...) gets curiosity for free.

* :class:`RND` — Random Network Distillation (Burda et al., 2018): the error of
  a predictor network trying to match a fixed random target network. Novel
  observations have high error (the predictor has not seen them), so the bonus
  decays as a state is revisited.
* :class:`ICM` — Intrinsic Curiosity Module (Pathak et al., 2017): the error of a
  forward dynamics model in a *learned* feature space that is trained (via an
  inverse model) to encode only what the agent can control, ignoring
  uncontrollable distractors.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from ..core.env import Env, Wrapper
from ..core.spaces import is_discrete
from ..networks.mlp import build_mlp
from ..utils.running_mean_std import RunningMeanStd
from ..utils.torch_utils import get_device, to_tensor

__all__ = ["IntrinsicRewardModule", "RND", "ICM", "CuriosityWrapper"]


class IntrinsicRewardModule(nn.Module):
    """Interface for intrinsic-reward generators used by :class:`CuriosityWrapper`.

    Implementations score a transition (higher = more novel/surprising) and train
    their own parameters from the same transitions.
    """

    def intrinsic_reward(self, obs, action, next_obs) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def update(self, obs, action, next_obs) -> float:  # pragma: no cover - interface
        raise NotImplementedError


def _obs2d(x, device) -> torch.Tensor:
    return to_tensor(np.asarray(x, dtype=np.float32).reshape(1, -1), device)


class RND(IntrinsicRewardModule):
    """Random Network Distillation intrinsic reward."""

    def __init__(
        self,
        obs_dim: int,
        feature_dim: int = 64,
        hidden_sizes=(128, 128),
        learning_rate: float = 1e-3,
        device: str = "auto",
    ) -> None:
        super().__init__()
        self.device = get_device(device)
        self.target = build_mlp(obs_dim, feature_dim, hidden_sizes).to(self.device)
        self.predictor = build_mlp(obs_dim, feature_dim, hidden_sizes).to(self.device)
        # The target is a fixed random projection; only the predictor learns.
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=learning_rate)

    @torch.no_grad()
    def intrinsic_reward(self, obs, action, next_obs) -> float:
        n = _obs2d(next_obs, self.device)
        err = ((self.predictor(n) - self.target(n)) ** 2).mean().item()
        return float(err)

    def update(self, obs, action, next_obs) -> float:
        n = _obs2d(next_obs, self.device)
        target = self.target(n).detach()
        loss = ((self.predictor(n) - target) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())


class ICM(IntrinsicRewardModule):
    """Intrinsic Curiosity Module (forward-model error in a learned feature space)."""

    def __init__(
        self,
        obs_dim: int,
        action_space,
        feature_dim: int = 64,
        hidden_sizes=(128, 128),
        learning_rate: float = 1e-3,
        beta: float = 0.2,
        device: str = "auto",
    ) -> None:
        super().__init__()
        self.device = get_device(device)
        self.beta = float(beta)
        self.discrete = is_discrete(action_space)
        self.action_dim = int(action_space.n) if self.discrete else int(action_space.shape[0])

        self.encoder = build_mlp(obs_dim, feature_dim, hidden_sizes).to(self.device)
        # Inverse model: predict the action that took phi(s) -> phi(s').
        self.inverse = build_mlp(2 * feature_dim, self.action_dim, hidden_sizes).to(self.device)
        # Forward model: predict phi(s') from phi(s) and the action.
        self.forward_model = build_mlp(
            feature_dim + self.action_dim, feature_dim, hidden_sizes
        ).to(self.device)
        self.feature_dim = feature_dim
        params = (
            list(self.encoder.parameters())
            + list(self.inverse.parameters())
            + list(self.forward_model.parameters())
        )
        self.optimizer = torch.optim.Adam(params, lr=learning_rate)
        self._ce = nn.CrossEntropyLoss()

    def _encode_action(self, action) -> torch.Tensor:
        if self.discrete:
            a = int(np.asarray(action).reshape(-1)[0])
            onehot = torch.zeros(1, self.action_dim, device=self.device)
            onehot[0, a] = 1.0
            return onehot
        return to_tensor(np.asarray(action, dtype=np.float32).reshape(1, -1), self.device)

    @torch.no_grad()
    def intrinsic_reward(self, obs, action, next_obs) -> float:
        phi = self.encoder(_obs2d(obs, self.device))
        phi_next = self.encoder(_obs2d(next_obs, self.device))
        a = self._encode_action(action)
        pred_next = self.forward_model(torch.cat([phi, a], dim=-1))
        return float(((pred_next - phi_next) ** 2).mean().item())

    def update(self, obs, action, next_obs) -> float:
        phi = self.encoder(_obs2d(obs, self.device))
        phi_next = self.encoder(_obs2d(next_obs, self.device))
        a = self._encode_action(action)

        pred_next = self.forward_model(torch.cat([phi, a.detach()], dim=-1))
        forward_loss = ((pred_next - phi_next.detach()) ** 2).mean()

        pred_action = self.inverse(torch.cat([phi, phi_next], dim=-1))
        if self.discrete:
            target = torch.tensor([int(np.asarray(action).reshape(-1)[0])], device=self.device)
            inverse_loss = self._ce(pred_action, target)
        else:
            inverse_loss = ((pred_action - a) ** 2).mean()

        loss = self.beta * forward_loss + (1.0 - self.beta) * inverse_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())


class CuriosityWrapper(Wrapper):
    """Add an intrinsic-reward bonus (from an :class:`IntrinsicRewardModule`).

    The wrapped ``step`` returns ``extrinsic + intrinsic_coef * normalized_bonus``.
    The raw components are exposed in ``info["extrinsic_reward"]`` and
    ``info["intrinsic_reward"]`` so evaluation can still measure true performance.
    By default the module is trained online from each observed transition.
    """

    def __init__(
        self,
        env: Env,
        module: IntrinsicRewardModule,
        intrinsic_coef: float = 1.0,
        normalize: bool = True,
        train: bool = True,
    ) -> None:
        super().__init__(env)
        self.module = module
        self.intrinsic_coef = float(intrinsic_coef)
        self.normalize = bool(normalize)
        self.train = bool(train)
        self._rms = RunningMeanStd(shape=(1,)) if normalize else None
        self._last_obs: Optional[np.ndarray] = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_obs = np.asarray(obs, dtype=np.float32)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        next_obs = np.asarray(obs, dtype=np.float32)

        intrinsic = self.module.intrinsic_reward(self._last_obs, action, next_obs)
        if self.train:
            self.module.update(self._last_obs, action, next_obs)

        bonus = intrinsic
        if self.normalize:
            self._rms.update(np.array([[intrinsic]], dtype=np.float64))
            bonus = intrinsic / float(np.sqrt(self._rms.var[0] + 1e-8))

        info = dict(info)
        info["extrinsic_reward"] = float(reward)
        info["intrinsic_reward"] = float(intrinsic)
        total = float(reward) + self.intrinsic_coef * bonus

        self._last_obs = next_obs
        return obs, total, terminated, truncated, info
