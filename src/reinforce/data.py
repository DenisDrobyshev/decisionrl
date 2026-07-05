"""Offline datasets of transitions for offline / batch RL.

A :class:`TransitionDataset` is a fixed set of ``(obs, action, reward, next_obs,
done)`` tuples that offline algorithms (e.g. :class:`~reinforce.algorithms.TD3BC`)
learn from without any further environment interaction. :func:`collect_dataset`
records one by rolling out a behaviour policy.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch

from .buffers.replay import ReplayBatch
from .core.env import Env

__all__ = ["TransitionDataset", "collect_dataset"]


class TransitionDataset:
    def __init__(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
        gamma: float = 0.99,
        device: str = "cpu",
        seed: Optional[int] = None,
    ) -> None:
        self.obs = np.asarray(obs, dtype=np.float32)
        self.actions = np.asarray(actions, dtype=np.float32)
        self.rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
        self.next_obs = np.asarray(next_obs, dtype=np.float32)
        self.dones = np.asarray(dones, dtype=np.float32).reshape(-1)
        self.gamma = float(gamma)
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        assert len(self.obs) == len(self.actions) == len(self.rewards)

    def __len__(self) -> int:
        return len(self.obs)

    def sample(self, batch_size: int) -> ReplayBatch:
        idx = self.rng.integers(0, len(self), size=batch_size)
        t = lambda a, d=torch.float32: torch.as_tensor(a[idx], device=self.device, dtype=d)  # noqa: E731
        return ReplayBatch(
            obs=t(self.obs),
            actions=t(self.actions),
            rewards=t(self.rewards),
            next_obs=t(self.next_obs),
            dones=t(self.dones),
            discounts=torch.full((batch_size,), self.gamma, device=self.device),
        )

    def normalize_obs(self, eps: float = 1e-3):
        """Return ``(mean, std)`` of observations and standardize in place."""
        mean = self.obs.mean(axis=0)
        std = self.obs.std(axis=0) + eps
        self.obs = (self.obs - mean) / std
        self.next_obs = (self.next_obs - mean) / std
        return mean, std


def collect_dataset(
    env: Env,
    policy: Callable[[np.ndarray], np.ndarray],
    n_transitions: int,
    gamma: float = 0.99,
    seed: Optional[int] = None,
    device: str = "cpu",
) -> TransitionDataset:
    """Roll out ``policy`` in ``env`` and record ``n_transitions`` transitions."""
    obs_buf, act_buf, rew_buf, next_buf, done_buf = [], [], [], [], []
    obs, _ = env.reset(seed=seed)
    for _ in range(n_transitions):
        action = policy(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        obs_buf.append(np.asarray(obs, dtype=np.float32))
        act_buf.append(np.asarray(action, dtype=np.float32))
        rew_buf.append(reward)
        next_buf.append(np.asarray(next_obs, dtype=np.float32))
        done_buf.append(terminated)
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
    return TransitionDataset(
        np.array(obs_buf), np.array(act_buf), np.array(rew_buf),
        np.array(next_buf), np.array(done_buf), gamma=gamma, device=device, seed=seed,
    )
