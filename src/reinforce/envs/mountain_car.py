"""Mountain Car (discrete and continuous), classic-control tasks from scratch.

Dynamics match ``MountainCar-v0`` / ``MountainCarContinuous-v0``: an
under-powered car must build momentum to escape a valley. The discrete version
has a sparse -1-per-step reward (a hard exploration problem); the continuous
version adds a control-cost penalty and a bonus for reaching the goal.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

from ..core.env import Env
from ..core.spaces import Box, Discrete

__all__ = ["MountainCar", "MountainCarContinuous"]


class _MountainCarBase(Env):
    min_position = -1.2
    max_position = 0.6
    max_speed = 0.07
    goal_position = 0.5
    gravity = 0.0025

    def __init__(self, max_steps: int) -> None:
        self.max_steps = int(max_steps)
        high = np.array([self.max_position, self.max_speed], dtype=np.float32)
        low = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        self.observation_space = Box(low, high, dtype=np.float32)
        self._rng = np.random.default_rng()
        self._pos = 0.0
        self._vel = 0.0
        self._steps = 0

    def _obs(self) -> np.ndarray:
        return np.array([self._pos, self._vel], dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._pos = float(self._rng.uniform(-0.6, -0.4))
        self._vel = 0.0
        self._steps = 0
        return self._obs(), {}

    @staticmethod
    def _height(x):
        return np.sin(3 * x) * 0.45 + 0.55

    def render_rgb(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from ..utils.render import fig_to_rgb

        fig, ax = plt.subplots(figsize=(4, 3), dpi=64)
        xs = np.linspace(self.min_position, self.max_position, 100)
        ax.plot(xs, self._height(xs), color="#475569", lw=2)
        ax.plot([self.goal_position], [self._height(self.goal_position)], marker="^", color="#16a34a", ms=12)
        ax.plot([self._pos], [self._height(self._pos)], marker="o", color="#2563eb", ms=12)
        ax.set_xlim(self.min_position, self.max_position)
        ax.set_ylim(0, 1.2)
        ax.axis("off")
        frame = fig_to_rgb(fig)
        plt.close(fig)
        return frame


class MountainCar(_MountainCarBase):
    force = 0.001

    def __init__(self, max_steps: int = 200) -> None:
        super().__init__(max_steps)
        self.action_space = Discrete(3)  # 0 push left, 1 no push, 2 push right

    def step(self, action: int):
        action = int(action)
        assert self.action_space.contains(action), f"invalid action {action}"
        self._vel += (action - 1) * self.force + math.cos(3 * self._pos) * (-self.gravity)
        self._vel = float(np.clip(self._vel, -self.max_speed, self.max_speed))
        self._pos += self._vel
        self._pos = float(np.clip(self._pos, self.min_position, self.max_position))
        if self._pos == self.min_position and self._vel < 0:
            self._vel = 0.0

        self._steps += 1
        terminated = bool(self._pos >= self.goal_position)
        truncated = self._steps >= self.max_steps and not terminated
        return self._obs(), -1.0, terminated, truncated, {}


class MountainCarContinuous(_MountainCarBase):
    power = 0.0015

    def __init__(self, max_steps: int = 999) -> None:
        super().__init__(max_steps)
        self.action_space = Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def step(self, action):
        force = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        self._vel += force * self.power - self.gravity * math.cos(3 * self._pos)
        self._vel = float(np.clip(self._vel, -self.max_speed, self.max_speed))
        self._pos += self._vel
        self._pos = float(np.clip(self._pos, self.min_position, self.max_position))
        if self._pos == self.min_position and self._vel < 0:
            self._vel = 0.0

        self._steps += 1
        terminated = bool(self._pos >= self.goal_position)
        reward = -0.1 * force ** 2 + (100.0 if terminated else 0.0)
        truncated = self._steps >= self.max_steps and not terminated
        return self._obs(), float(reward), terminated, truncated, {}
