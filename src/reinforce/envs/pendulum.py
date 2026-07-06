"""Pendulum swing-up: a continuous-control classic-control task from scratch.

Matches the dynamics of ``Pendulum-v1``. Observation is
``[cos(theta), sin(theta), theta_dot]`` and the action is a 1-D torque in
``[-max_torque, max_torque]``. The reward penalizes angle, velocity and torque,
so it is always negative and maximized (towards 0) by balancing upright.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..core.env import Env
from ..core.spaces import Box

__all__ = ["Pendulum"]


class Pendulum(Env):
    max_speed = 8.0
    max_torque = 2.0
    dt = 0.05
    gravity = 10.0
    mass = 1.0
    length = 1.0

    def __init__(self, max_steps: int = 200) -> None:
        self.max_steps = int(max_steps)
        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        self.observation_space = Box(-high, high, dtype=np.float32)
        self.action_space = Box(-self.max_torque, self.max_torque, shape=(1,), dtype=np.float32)
        self._rng = np.random.default_rng()
        self._theta = 0.0
        self._theta_dot = 0.0
        self._steps = 0

    def _obs(self) -> np.ndarray:
        return np.array(
            [np.cos(self._theta), np.sin(self._theta), self._theta_dot], dtype=np.float32
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._theta = float(self._rng.uniform(-np.pi, np.pi))
        self._theta_dot = float(self._rng.uniform(-1.0, 1.0))
        self._steps = 0
        return self._obs(), {}

    def step(self, action):
        u = float(np.clip(np.asarray(action).reshape(-1)[0], -self.max_torque, self.max_torque))
        g, m, length, dt = self.gravity, self.mass, self.length, self.dt

        # angle normalized to [-pi, pi] for the cost
        theta_norm = ((self._theta + np.pi) % (2 * np.pi)) - np.pi
        cost = theta_norm**2 + 0.1 * self._theta_dot**2 + 0.001 * u**2

        new_theta_dot = (
            self._theta_dot
            + (3 * g / (2 * length) * np.sin(self._theta) + 3.0 / (m * length**2) * u) * dt
        )
        new_theta_dot = float(np.clip(new_theta_dot, -self.max_speed, self.max_speed))
        new_theta = self._theta + new_theta_dot * dt

        self._theta = new_theta
        self._theta_dot = new_theta_dot
        self._steps += 1
        truncated = self._steps >= self.max_steps
        return self._obs(), float(-cost), False, truncated, {}

    def render_rgb(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from ..utils.render import fig_to_rgb

        fig, ax = plt.subplots(figsize=(3, 3), dpi=72)
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect("equal")
        ax.axis("off")
        tip = (np.sin(self._theta), np.cos(self._theta))  # theta=0 is upright
        ax.plot([0, tip[0]], [0, tip[1]], color="#db2777", lw=6, solid_capstyle="round")
        ax.plot([0], [0], marker="o", color="#1e293b", ms=8)
        ax.plot([tip[0]], [tip[1]], marker="o", color="#db2777", ms=10)
        frame = fig_to_rgb(fig)
        plt.close(fig)
        return frame
