"""Battery energy arbitrage in a microgrid: an applied-RL energy problem.

A battery sits between a time-varying electricity price, local (solar-like)
generation and a household load. Each step the agent charges or discharges the
battery; energy drawn from the grid is bought at the current price and surplus is
sold back at the same price. The optimal policy stores cheap/surplus energy and
discharges it during expensive peaks -- price arbitrage plus self-consumption.
Doing nothing (always buy at spot) is the baseline to beat.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..core.env import Env
from ..core.spaces import Box

__all__ = ["EnergyMicrogrid"]


class EnergyMicrogrid(Env):
    def __init__(
        self,
        capacity: float = 10.0,
        max_power: float = 3.0,
        efficiency: float = 0.95,
        horizon: int = 48,  # half-hour steps over a day
    ) -> None:
        self.capacity = float(capacity)
        self.max_power = float(max_power)
        self.efficiency = float(efficiency)
        self.horizon = int(horizon)

        # Observation: state-of-charge, price, generation, load (all normalized), plus
        # a (sin, cos) clock so the agent can anticipate the daily cycle.
        self.observation_space = Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        self.action_space = Box(-1.0, 1.0, shape=(1,), dtype=np.float32)  # charge(+)/discharge(-)

        self._rng = np.random.default_rng()
        self._soc = 0.5 * self.capacity
        self._steps = 0

    def _price(self, t: int) -> float:
        phase = 2 * np.pi * t / self.horizon
        # Cheap overnight, expensive in the evening peak.
        return float(0.6 + 0.4 * np.sin(phase - np.pi / 2) + 0.05 * self._rng.standard_normal())

    def _generation(self, t: int) -> float:
        # Solar: zero at night, peak at midday.
        solar = np.sin(np.pi * t / self.horizon)
        return float(max(0.0, 3.0 * solar + 0.1 * self._rng.standard_normal()))

    def _load(self, t: int) -> float:
        phase = 2 * np.pi * t / self.horizon
        return float(1.5 + 0.8 * max(0.0, np.sin(phase - np.pi / 2)) + 0.1 * self._rng.standard_normal())

    def _obs(self, price: float, gen: float, load: float) -> np.ndarray:
        phase = 2 * np.pi * self._steps / self.horizon
        return np.array(
            [
                2.0 * self._soc / self.capacity - 1.0,
                np.clip(price, 0.0, 1.5) / 1.5,
                np.clip(gen, 0.0, 4.0) / 4.0,
                np.clip(load, 0.0, 4.0) / 4.0,
                np.sin(phase),
                np.cos(phase),
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._soc = 0.5 * self.capacity
        self._steps = 0
        price, gen, load = self._price(0), self._generation(0), self._load(0)
        return self._obs(price, gen, load), {}

    def step(self, action):
        a = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        power = a * self.max_power  # kW over the step (positive = charge)

        # Clip to what the battery can physically absorb/supply this step.
        if power >= 0:
            power = min(power, (self.capacity - self._soc) / self.efficiency)
            self._soc += power * self.efficiency
        else:
            power = -min(-power, self._soc * self.efficiency)
            self._soc += power / self.efficiency
        self._soc = float(np.clip(self._soc, 0.0, self.capacity))

        price = self._price(self._steps)
        gen = self._generation(self._steps)
        load = self._load(self._steps)
        grid = load - gen + power  # net draw from the grid (may be negative = export)
        reward = -price * grid  # cost of buying (or revenue from selling)

        self._steps += 1
        truncated = self._steps >= self.horizon
        info = {"price": price, "generation": gen, "load": load, "soc": self._soc, "grid": grid}
        return self._obs(price, gen, load), float(reward), False, truncated, info
