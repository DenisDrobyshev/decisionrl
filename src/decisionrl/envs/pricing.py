"""Dynamic pricing / revenue management: a classic applied-RL operations problem.

A finite stock of a perishable good must be sold over a fixed selling horizon.
Each period the agent sets a price; demand is price-elastic and stochastic, so a
high price earns more per unit but risks unsold stock at the deadline, while a low
price sells out early and leaves money on the table. The optimal policy raises the
price when inventory is scarce relative to the time remaining -- exactly the
behaviour airlines and hotels use. A fixed price is the baseline to beat.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..core.env import Env
from ..core.spaces import Box, Discrete

__all__ = ["DynamicPricing"]


class DynamicPricing(Env):
    def __init__(
        self,
        n_prices: int = 8,
        price_min: float = 0.5,
        price_max: float = 4.0,
        initial_inventory: int = 8,
        base_demand: float = 8.0,
        elasticity: float = 1.0,
        horizon: int = 20,
    ) -> None:
        self.prices = np.linspace(float(price_min), float(price_max), int(n_prices))
        self.price_min = float(price_min)
        self.price_max = float(price_max)
        self.initial_inventory = int(initial_inventory)
        self.base_demand = float(base_demand)
        self.elasticity = float(elasticity)
        self.horizon = int(horizon)

        # Observation: remaining inventory and remaining time, both scaled to [0, 1].
        self.observation_space = Box(0.0, 1.0, shape=(2,), dtype=np.float32)
        self.action_space = Discrete(len(self.prices))

        self._rng = np.random.default_rng()
        self._inventory = self.initial_inventory
        self._steps = 0

    def _obs(self) -> np.ndarray:
        return np.array(
            [self._inventory / self.initial_inventory, 1.0 - self._steps / self.horizon],
            dtype=np.float32,
        )

    def _demand_mean(self, price: float) -> float:
        # Expected demand falls exponentially as price rises above the floor.
        return self.base_demand * float(np.exp(-self.elasticity * (price - self.price_min)))

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._inventory = self.initial_inventory
        self._steps = 0
        return self._obs(), {}

    def step(self, action: int):
        price = float(self.prices[int(action)])
        demand = int(self._rng.poisson(self._demand_mean(price)))
        sales = min(self._inventory, demand)
        self._inventory -= sales
        reward = price * sales

        self._steps += 1
        terminated = self._inventory <= 0
        truncated = self._steps >= self.horizon
        info = {"price": price, "demand": demand, "sales": sales, "inventory": self._inventory}
        return self._obs(), float(reward), terminated, truncated, info
