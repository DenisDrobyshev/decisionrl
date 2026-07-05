"""A tiny continuous-control reaching task.

A point mass in a 2-D box must reach the origin. The reward is dense
(``-distance``), the dynamics are trivial, and an optimal policy is easy to
learn, which makes this the fast smoke/learning test-bed for continuous-control
algorithms (DDPG, TD3, SAC, continuous PPO).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..core.env import Env
from ..core.spaces import Box

__all__ = ["PointMass"]


class PointMass(Env):
    def __init__(self, dim: int = 2, max_steps: int = 50, dt: float = 0.1, goal_radius: float = 0.05) -> None:
        self.dim = int(dim)
        self.max_steps = int(max_steps)
        self.dt = float(dt)
        self.goal_radius = float(goal_radius)

        self.observation_space = Box(-1.0, 1.0, shape=(self.dim,), dtype=np.float32)
        self.action_space = Box(-1.0, 1.0, shape=(self.dim,), dtype=np.float32)

        self._rng = np.random.default_rng()
        self._pos = np.zeros(self.dim, dtype=np.float32)
        self._steps = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._pos = self._rng.uniform(-1.0, 1.0, size=self.dim).astype(np.float32)
        self._steps = 0
        return self._pos.copy(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(self.dim), -1.0, 1.0)
        self._pos = np.clip(self._pos + self.dt * action, -1.0, 1.0).astype(np.float32)
        self._steps += 1

        distance = float(np.linalg.norm(self._pos))
        reward = -distance
        terminated = distance < self.goal_radius
        truncated = self._steps >= self.max_steps and not terminated
        return self._pos.copy(), reward, terminated, truncated, {"distance": distance}
