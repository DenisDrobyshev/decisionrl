"""Prioritized experience replay (Schaul et al., 2016) with a sum-tree."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from ..core.spaces import Space
from .replay import ReplayBatch, ReplayBuffer

__all__ = ["PrioritizedReplayBuffer", "SumTree"]


class SumTree:
    """A binary tree where each parent is the sum of its children.

    Enables O(log n) proportional sampling and priority updates.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.tree = np.zeros(2 * self.capacity - 1, dtype=np.float64)

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def update(self, data_idx: int, priority: float) -> None:
        tree_idx = data_idx + self.capacity - 1
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get(self, value: float) -> int:
        """Return the data index whose cumulative range contains ``value``."""
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                break
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right
        return idx - (self.capacity - 1)


class PrioritizedReplayBuffer(ReplayBuffer):
    """Replay buffer that samples transitions in proportion to their TD-error.

    New transitions get the current maximum priority so they are seen at least
    once. Importance-sampling weights correct for the non-uniform sampling and
    are annealed via ``beta`` -> 1.
    """

    def __init__(
        self,
        capacity: int,
        observation_space: Space,
        action_space: Space,
        alpha: float = 0.6,
        beta: float = 0.4,
        epsilon: float = 1e-6,
        device: str = "cpu",
        seed: Optional[int] = None,
        n_step: int = 1,
        gamma: float = 0.99,
    ) -> None:
        super().__init__(
            capacity, observation_space, action_space,
            device=device, seed=seed, n_step=n_step, gamma=gamma,
        )
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.epsilon = float(epsilon)
        self.tree = SumTree(self.capacity)
        self.max_priority = 1.0

    def _store_transition(self, obs, action, reward, next_obs, done, discount) -> int:
        idx = super()._store_transition(obs, action, reward, next_obs, done, discount)
        self.tree.update(idx, self.max_priority ** self.alpha)
        return idx

    def sample(self, batch_size: int, beta: Optional[float] = None) -> ReplayBatch:
        assert self.size > 0, "cannot sample from an empty buffer"
        beta = self.beta if beta is None else beta
        total = self.tree.total
        segment = total / batch_size

        indices = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            value = self.rng.uniform(a, b)
            data_idx = self.tree.get(value)
            data_idx = min(data_idx, self.size - 1)  # guard against edge float error
            indices[i] = data_idx
            priorities[i] = self.tree.tree[data_idx + self.capacity - 1]

        probs = priorities / total
        weights = (self.size * probs) ** (-beta)
        weights = weights / weights.max()

        batch = self._to_batch(indices)
        batch.weights = torch.as_tensor(weights, device=self.device, dtype=torch.float32)
        batch.indices = indices
        return batch

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        priorities = np.abs(td_errors) + self.epsilon
        self.max_priority = max(self.max_priority, float(priorities.max()))
        for idx, priority in zip(indices, priorities):
            self.tree.update(int(idx), float(priority) ** self.alpha)
