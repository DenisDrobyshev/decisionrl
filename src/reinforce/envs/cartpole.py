"""CartPole classic-control task, implemented from scratch.

Dynamics match the canonical Barto/Sutton/Anderson cart-pole (the same as
``CartPole-v1``), so agents validated here transfer directly to Gymnasium.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

from ..core.env import Env
from ..core.spaces import Box, Discrete

__all__ = ["CartPole"]


class CartPole(Env):
    """Balance a pole on a cart by pushing left (0) or right (1).

    Reward is +1 per step. The episode terminates when the pole falls beyond
    +-12 degrees or the cart leaves +-2.4 units, and truncates at
    ``max_steps`` (500 by default, as in ``CartPole-v1``).
    """

    metadata = {"render_modes": []}

    gravity = 9.8
    mass_cart = 1.0
    mass_pole = 0.1
    length = 0.5  # half the pole length
    force_mag = 10.0
    tau = 0.02  # seconds between state updates

    def __init__(self, max_steps: int = 500) -> None:
        self.max_steps = int(max_steps)
        self.total_mass = self.mass_cart + self.mass_pole
        self.pole_mass_length = self.mass_pole * self.length

        self.x_threshold = 2.4
        self.theta_threshold = 12 * 2 * math.pi / 360

        high = np.array(
            [self.x_threshold * 2, np.inf, self.theta_threshold * 2, np.inf],
            dtype=np.float32,
        )
        self.observation_space = Box(-high, high, dtype=np.float32)
        self.action_space = Discrete(2)

        self._rng = np.random.default_rng()
        self._state: Optional[np.ndarray] = None
        self._steps = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._state = self._rng.uniform(-0.05, 0.05, size=(4,)).astype(np.float32)
        self._steps = 0
        return self._state.copy(), {}

    def step(self, action: int):
        assert self._state is not None, "call reset() before step()"
        action = int(action)
        assert self.action_space.contains(action), f"invalid action {action}"

        x, x_dot, theta, theta_dot = self._state
        force = self.force_mag if action == 1 else -self.force_mag
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        temp = (force + self.pole_mass_length * theta_dot**2 * sin_t) / self.total_mass
        theta_acc = (self.gravity * sin_t - cos_t * temp) / (
            self.length * (4.0 / 3.0 - self.mass_pole * cos_t**2 / self.total_mass)
        )
        x_acc = temp - self.pole_mass_length * theta_acc * cos_t / self.total_mass

        # semi-implicit Euler integration (as in Gymnasium)
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * x_acc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * theta_acc
        self._state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

        self._steps += 1
        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold
            or theta > self.theta_threshold
        )
        truncated = self._steps >= self.max_steps and not terminated
        reward = 1.0
        return self._state.copy(), reward, terminated, truncated, {}
