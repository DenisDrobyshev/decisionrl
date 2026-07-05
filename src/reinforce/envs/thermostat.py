"""Thermostat / HVAC control: an applied continuous-control problem.

The agent modulates a heating/cooling unit to keep an indoor temperature close
to a setpoint while a periodically varying outdoor temperature pulls it away.
The reward penalizes both comfort violations and energy use, so a good policy
tracks the setpoint smoothly with minimal power - which a naive bang-bang
thermostat cannot do.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..core.env import Env
from ..core.spaces import Box

__all__ = ["Thermostat"]


class Thermostat(Env):
    def __init__(
        self,
        setpoint: float = 21.0,
        outdoor_mean: float = 5.0,
        outdoor_amplitude: float = 5.0,
        outdoor_period: float = 100.0,
        leak: float = 0.05,
        heat_coef: float = 1.5,
        energy_weight: float = 0.1,
        horizon: int = 200,
        noise: float = 0.3,
    ) -> None:
        self.setpoint = float(setpoint)
        self.outdoor_mean = float(outdoor_mean)
        self.outdoor_amplitude = float(outdoor_amplitude)
        self.outdoor_period = float(outdoor_period)
        self.leak = float(leak)
        self.heat_coef = float(heat_coef)
        self.energy_weight = float(energy_weight)
        self.horizon = int(horizon)
        self.noise = float(noise)

        # obs = [ (indoor - setpoint)/10, (outdoor - setpoint)/20 ]
        self.observation_space = Box(-5.0, 5.0, shape=(2,), dtype=np.float32)
        # action = heating/cooling power in [-1, 1]
        self.action_space = Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        self._rng = np.random.default_rng()
        self._indoor = setpoint
        self._steps = 0

    def _outdoor(self, t: int) -> float:
        base = self.outdoor_mean + self.outdoor_amplitude * np.sin(2 * np.pi * t / self.outdoor_period)
        return base + self.noise * self._rng.standard_normal()

    def _obs(self, outdoor: float) -> np.ndarray:
        obs = np.array(
            [(self._indoor - self.setpoint) / 10.0, (outdoor - self.setpoint) / 20.0],
            dtype=np.float32,
        )
        return np.clip(obs, -5.0, 5.0)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._indoor = self.setpoint + self._rng.uniform(-3.0, 3.0)
        self._steps = 0
        self._outdoor_now = self._outdoor(0)
        return self._obs(self._outdoor_now), {}

    def step(self, action):
        power = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        outdoor = self._outdoor_now

        # thermal dynamics: leak toward outdoor + heating/cooling input
        self._indoor = self._indoor + self.leak * (outdoor - self._indoor) + self.heat_coef * power

        self._steps += 1
        self._outdoor_now = self._outdoor(self._steps)

        comfort_error = self._indoor - self.setpoint
        reward = -(comfort_error ** 2 + self.energy_weight * power ** 2)

        truncated = self._steps >= self.horizon
        info = {"indoor": self._indoor, "outdoor": outdoor, "power": power}
        return self._obs(self._outdoor_now), float(reward), False, truncated, info
