"""Lunar lander — a self-contained rocket soft-landing control task.

A simplified 2-D rigid-body lander (in the spirit of ``LunarLander-v2``) must
descend under gravity and touch down gently, upright, on a central pad using
three thrusters (left / main / right). The 8-D state and shaped, potential-based
reward (distance, speed, tilt, leg contact, fuel use) with large terminal bonus
/ penalty make credit assignment non-trivial. Discrete 4-action control — solve
with PPO or DQN.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

from ..core.env import Env
from ..core.spaces import Box, Discrete

__all__ = ["LunarLander"]


class LunarLander(Env):
    metadata = {"render_modes": []}

    gravity = 0.4
    main_thrust = 1.0
    side_thrust = 0.08
    side_torque = 2.0
    dt = 0.05

    def __init__(self, max_steps: int = 300) -> None:
        self.max_steps = int(max_steps)
        self.pad_half = 0.2
        # state: x, y, vx, vy, angle, angular_vel, left_contact, right_contact
        # (velocity/angle are left unbounded, as in LunarLander-v2)
        self.observation_space = Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)
        self.action_space = Discrete(4)  # 0 nop, 1 left, 2 main, 3 right

        self._rng = np.random.default_rng()
        self._state = np.zeros(8, dtype=np.float64)
        self._steps = 0
        self._prev_shaping: Optional[float] = None

    def _shaping(self, s: np.ndarray) -> float:
        x, y, vx, vy, angle = s[0], s[1], s[2], s[3], s[4]
        legs = s[6] + s[7]
        return (
            -100.0 * math.sqrt(x * x + y * y)
            - 100.0 * math.sqrt(vx * vx + vy * vy)
            - 100.0 * abs(angle)
            + 10.0 * legs
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._state = np.array(
            [
                self._rng.uniform(-0.3, 0.3), 1.3,
                self._rng.uniform(-0.1, 0.1), self._rng.uniform(-0.1, 0.0),
                self._rng.uniform(-0.1, 0.1), 0.0,
                0.0, 0.0,
            ],
            dtype=np.float64,
        )
        self._steps = 0
        self._prev_shaping = self._shaping(self._state)
        return self._state.astype(np.float32), {}

    def step(self, action):
        action = int(action)
        x, y, vx, vy, angle, ang_vel, _, _ = self._state

        ax, ay, torque = 0.0, -self.gravity, 0.0
        if action == 2:  # main engine: thrust along body-up
            ax += -math.sin(angle) * self.main_thrust
            ay += math.cos(angle) * self.main_thrust
        elif action == 1:  # left engine: rotate + slight push
            torque += self.side_torque
            ax += math.cos(angle) * self.side_thrust
        elif action == 3:  # right engine
            torque -= self.side_torque
            ax -= math.cos(angle) * self.side_thrust

        vx += ax * self.dt
        vy += ay * self.dt
        ang_vel += torque * self.dt
        x += vx * self.dt
        y += vy * self.dt
        angle += ang_vel * self.dt

        left_contact = 1.0 if y < 0.1 else 0.0
        right_contact = left_contact
        self._state = np.array([x, y, vx, vy, angle, ang_vel, left_contact, right_contact])

        self._steps += 1
        shaping = self._shaping(self._state)
        reward = shaping - (self._prev_shaping if self._prev_shaping is not None else shaping)
        self._prev_shaping = shaping
        if action == 2:
            reward -= 0.3
        elif action in (1, 3):
            reward -= 0.03

        terminated = False
        truncated = False
        if y <= 0.0:  # touchdown
            terminated = True
            gentle = abs(vx) < 0.4 and abs(vy) < 0.5 and abs(angle) < 0.25
            on_pad = abs(x) < self.pad_half
            reward += 100.0 if (gentle and on_pad) else -100.0
        elif abs(x) > 1.2 or y > 1.8:  # flew out of bounds
            terminated = True
            reward -= 100.0
        elif self._steps >= self.max_steps:
            truncated = True

        return self._state.astype(np.float32), float(reward), terminated, truncated, {}
