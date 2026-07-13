"""Acrobot: a two-link underactuated pendulum (classic control, from scratch).

Matches ``Acrobot-v1``: torque is applied only at the second joint and the goal
is to swing the tip above a line. Observation is
``[cos θ1, sin θ1, cos θ2, sin θ2, θ1_dot, θ2_dot]``; reward is -1 per step.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..core.env import Env
from ..core.spaces import Box, Discrete

__all__ = ["Acrobot"]


def _wrap(x: float, low: float, high: float) -> float:
    diff = high - low
    while x > high:
        x -= diff
    while x < low:
        x += diff
    return x


class Acrobot(Env):
    LINK_LENGTH_1 = 1.0
    LINK_MASS_1 = 1.0
    LINK_MASS_2 = 1.0
    LINK_COM_POS_1 = 0.5
    LINK_COM_POS_2 = 0.5
    LINK_MOI = 1.0
    MAX_VEL_1 = 4 * np.pi
    MAX_VEL_2 = 9 * np.pi
    G = 9.8
    dt = 0.2
    AVAIL_TORQUE = (-1.0, 0.0, 1.0)

    def __init__(self, max_steps: int = 500) -> None:
        self.max_steps = int(max_steps)
        high = np.array([1.0, 1.0, 1.0, 1.0, self.MAX_VEL_1, self.MAX_VEL_2], dtype=np.float32)
        self.observation_space = Box(-high, high, dtype=np.float32)
        self.action_space = Discrete(3)
        self._rng = np.random.default_rng()
        self._state = np.zeros(4, dtype=np.float64)
        self._steps = 0

    def _obs(self) -> np.ndarray:
        s = self._state
        return np.array(
            [np.cos(s[0]), np.sin(s[0]), np.cos(s[1]), np.sin(s[1]), s[2], s[3]], dtype=np.float32
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._state = self._rng.uniform(-0.1, 0.1, size=4)
        self._steps = 0
        return self._obs(), {}

    def _dsdt(self, s_augmented: np.ndarray) -> np.ndarray:
        m1, m2 = self.LINK_MASS_1, self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1, lc2 = self.LINK_COM_POS_1, self.LINK_COM_POS_2
        I1 = I2 = self.LINK_MOI
        g = self.G
        a = s_augmented[-1]
        theta1, theta2, dtheta1, dtheta2 = s_augmented[:4]
        d1 = m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * np.cos(theta2)) + I1 + I2
        d2 = m2 * (lc2**2 + l1 * lc2 * np.cos(theta2)) + I2
        phi2 = m2 * lc2 * g * np.cos(theta1 + theta2 - np.pi / 2.0)
        phi1 = (
            -m2 * l1 * lc2 * dtheta2**2 * np.sin(theta2)
            - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * np.sin(theta2)
            + (m1 * lc1 + m2 * l1) * g * np.cos(theta1 - np.pi / 2.0)
            + phi2
        )
        ddtheta2 = (a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1**2 * np.sin(theta2) - phi2) / (
            m2 * lc2**2 + I2 - d2**2 / d1
        )
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
        return np.array([dtheta1, dtheta2, ddtheta1, ddtheta2, 0.0])

    def _rk4(self, y0: np.ndarray) -> np.ndarray:
        dt = self.dt
        k1 = self._dsdt(y0)
        k2 = self._dsdt(y0 + dt / 2.0 * k1)
        k3 = self._dsdt(y0 + dt / 2.0 * k2)
        k4 = self._dsdt(y0 + dt * k3)
        return y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    def step(self, action: int):
        action = int(action)
        assert self.action_space.contains(action), f"invalid action {action}"
        torque = self.AVAIL_TORQUE[action]
        s_aug = np.append(self._state, torque)
        ns = self._rk4(s_aug)[:4]
        ns[0] = _wrap(ns[0], -np.pi, np.pi)
        ns[1] = _wrap(ns[1], -np.pi, np.pi)
        ns[2] = float(np.clip(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1))
        ns[3] = float(np.clip(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2))
        self._state = ns

        self._steps += 1
        terminated = bool(-np.cos(ns[0]) - np.cos(ns[1] + ns[0]) > 1.0)
        truncated = self._steps >= self.max_steps and not terminated
        return self._obs(), -1.0, terminated, truncated, {}
