"""Bit-flipping — the canonical sparse-reward, goal-conditioned benchmark.

The agent must turn an ``n``-bit state into a target ``n``-bit goal by flipping one
bit per step. Reward is ``0`` only when state == goal (else ``-1``), so for larger
``n`` the reward is so sparse that vanilla DQN never succeeds — but Hindsight
Experience Replay (:class:`~decisionrl.algorithms.HERDQN`) solves it easily by
relabelling failed episodes with the goals they *did* reach.

The observation is ``concat(state, goal)`` and ``compute_reward`` lets HER
recompute rewards for relabelled goals.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..core.env import Env
from ..core.spaces import Box, Discrete

__all__ = ["BitFlipping"]


class BitFlipping(Env):
    def __init__(self, n_bits: int = 8, max_steps: Optional[int] = None) -> None:
        self.n_bits = int(n_bits)
        self.max_steps = int(max_steps) if max_steps is not None else self.n_bits
        self.observation_space = Box(0.0, 1.0, shape=(2 * self.n_bits,), dtype=np.float32)
        self.action_space = Discrete(self.n_bits)
        self._rng = np.random.default_rng()
        self._state = np.zeros(self.n_bits, dtype=np.float32)
        self._goal = np.zeros(self.n_bits, dtype=np.float32)
        self._steps = 0

    @staticmethod
    def compute_reward(achieved: np.ndarray, desired: np.ndarray) -> float:
        return 0.0 if np.array_equal(achieved, desired) else -1.0

    def _obs(self) -> np.ndarray:
        return np.concatenate([self._state, self._goal]).astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._state = self._rng.integers(0, 2, self.n_bits).astype(np.float32)
        self._goal = self._rng.integers(0, 2, self.n_bits).astype(np.float32)
        while np.array_equal(self._state, self._goal):
            self._goal = self._rng.integers(0, 2, self.n_bits).astype(np.float32)
        self._steps = 0
        return self._obs(), {"achieved_goal": self._state.copy(), "desired_goal": self._goal.copy()}

    def step(self, action: int):
        self._state = self._state.copy()
        self._state[int(action)] = 1.0 - self._state[int(action)]  # flip the bit
        self._steps += 1
        reward = self.compute_reward(self._state, self._goal)
        terminated = reward == 0.0
        truncated = self._steps >= self.max_steps and not terminated
        info = {"achieved_goal": self._state.copy(), "desired_goal": self._goal.copy()}
        return self._obs(), reward, terminated, truncated, info
