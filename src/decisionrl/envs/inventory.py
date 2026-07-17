"""Inventory management: a classic applied RL / operations-research problem.

Each step the agent decides how many units to order to satisfy stochastic
(Poisson) customer demand, trading off holding cost, ordering cost and the lost
revenue / penalty of stockouts. The optimal solution is a base-stock ("order up
to S") policy, which a trained agent recovers.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..core.env import Env
from ..core.spaces import Box, Discrete

__all__ = ["InventoryManagement"]


class InventoryManagement(Env):
    def __init__(
        self,
        max_inventory: int = 20,
        max_order: int = 10,
        demand_mean: float = 5.0,
        price: float = 1.0,
        unit_cost: float = 0.3,
        holding_cost: float = 0.05,
        stockout_penalty: float = 0.2,
        horizon: int = 60,
    ) -> None:
        self.max_inventory = int(max_inventory)
        self.max_order = int(max_order)
        self.demand_mean = float(demand_mean)
        self.price = float(price)
        self.unit_cost = float(unit_cost)
        self.holding_cost = float(holding_cost)
        self.stockout_penalty = float(stockout_penalty)
        self.horizon = int(horizon)

        # Observation: current inventory level, scaled to [0, 1].
        self.observation_space = Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        self.action_space = Discrete(self.max_order + 1)  # order 0..max_order

        self._rng = np.random.default_rng()
        self._inventory = 0
        self._steps = 0

    def _obs(self) -> np.ndarray:
        return np.array([self._inventory / self.max_inventory], dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._inventory = int(self._rng.integers(0, self.max_inventory + 1))
        self._steps = 0
        return self._obs(), {}

    def step(self, action: int):
        order = int(np.clip(action, 0, self.max_order))
        inv_after_order = min(self._inventory + order, self.max_inventory)

        demand = int(self._rng.poisson(self.demand_mean))
        sales = min(inv_after_order, demand)
        lost = demand - sales
        self._inventory = inv_after_order - sales

        reward = (
            self.price * sales
            - self.unit_cost * order
            - self.holding_cost * self._inventory
            - self.stockout_penalty * lost
        )

        self._steps += 1
        truncated = self._steps >= self.horizon
        info = {"demand": demand, "sales": sales, "lost_sales": lost, "order": order}
        return self._obs(), float(reward), False, truncated, info

    def render_rgb(self):
        from ..utils.render import bars_frame
        return bars_frame(["inventory"], [self._inventory], self.max_inventory,
                          title="inventory level")
