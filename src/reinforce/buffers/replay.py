"""Uniform experience replay buffer for off-policy algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from ..core.spaces import Space, is_discrete

__all__ = ["ReplayBuffer", "ReplayBatch"]


@dataclass
class ReplayBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor
    weights: Optional[torch.Tensor] = None  # importance weights (PER)
    indices: Optional[np.ndarray] = None  # sampled indices (PER)


def _shape(space: Space):
    return space.shape if space.shape is not None else ()


class ReplayBuffer:
    """A ring buffer of transitions with uniform random sampling.

    ``done`` stored here is the *bootstrapping* terminal flag (Gymnasium's
    ``terminated``), never a time-limit truncation, so target computation stays
    correct.
    """

    def __init__(
        self,
        capacity: int,
        observation_space: Space,
        action_space: Space,
        device: str = "cpu",
        seed: Optional[int] = None,
    ) -> None:
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)

        self.discrete_actions = is_discrete(action_space)
        obs_shape = _shape(observation_space)
        obs_dtype = np.int64 if is_discrete(observation_space) else np.float32
        act_shape = () if self.discrete_actions else _shape(action_space)
        act_dtype = np.int64 if self.discrete_actions else np.float32

        self.obs = np.zeros((self.capacity, *obs_shape), dtype=obs_dtype)
        self.next_obs = np.zeros((self.capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.zeros((self.capacity, *act_shape), dtype=act_dtype)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)

        self.pos = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(self, obs, action, reward, next_obs, done) -> None:
        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _to_batch(self, idx: np.ndarray) -> ReplayBatch:
        return ReplayBatch(
            obs=torch.as_tensor(self.obs[idx], device=self.device, dtype=torch.float32),
            actions=torch.as_tensor(
                self.actions[idx],
                device=self.device,
                dtype=torch.long if self.discrete_actions else torch.float32,
            ),
            rewards=torch.as_tensor(self.rewards[idx], device=self.device, dtype=torch.float32),
            next_obs=torch.as_tensor(self.next_obs[idx], device=self.device, dtype=torch.float32),
            dones=torch.as_tensor(self.dones[idx], device=self.device, dtype=torch.float32),
        )

    def sample(self, batch_size: int) -> ReplayBatch:
        idx = self.rng.integers(0, self.size, size=batch_size)
        return self._to_batch(idx)
