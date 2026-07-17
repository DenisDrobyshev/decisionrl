"""Joint pricing + inventory control — coupled decisions with no closed form.

Each period the agent sets **both** the price and the replenishment order. The two
decisions are coupled: price sets demand, demand sets how much to order, and when
stock piles up the right move is to *mark the price down* to clear it rather than
eat holding cost. There is no clean closed-form joint optimum, and — crucially — no
single *static* (price, base-stock) rule is right, because the best price depends on
the current inventory. That is the gap a state-dependent learned policy exploits.

Continuous 2-D action (price, order); pairs with SAC / TD3 / PPO.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..core.env import Env
from ..core.spaces import Box

__all__ = ["JointPricingInventory"]


class JointPricingInventory(Env):
    def __init__(
        self,
        max_inventory: int = 30,
        max_order: int = 15,
        price_min: float = 0.5,
        price_max: float = 3.0,
        base_demand: float = 9.0,
        elasticity: float = 1.2,
        unit_cost: float = 0.3,
        holding_cost: float = 0.25,
        stockout_penalty: float = 0.3,
        horizon: int = 40,
    ) -> None:
        self.max_inventory = int(max_inventory)
        self.max_order = int(max_order)
        self.price_min = float(price_min)
        self.price_max = float(price_max)
        self.base_demand = float(base_demand)
        self.elasticity = float(elasticity)
        self.unit_cost = float(unit_cost)
        self.holding_cost = float(holding_cost)
        self.stockout_penalty = float(stockout_penalty)
        self.horizon = int(horizon)

        # Observation: inventory level and last demand (both scaled to [0, 1]).
        self.observation_space = Box(0.0, 1.0, shape=(2,), dtype=np.float32)
        # Action: [price, order], each in [-1, 1] (decoded to the real ranges).
        self.action_space = Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        self._rng = np.random.default_rng()
        self._inventory = 0
        self._last_demand = 0.0
        self._steps = 0

    def _demand_mean(self, price: float) -> float:
        return self.base_demand * float(np.exp(-self.elasticity * (price - self.price_min)))

    def _obs(self) -> np.ndarray:
        return np.array(
            [self._inventory / self.max_inventory,
             min(self._last_demand / self.base_demand, 1.0)],
            dtype=np.float32,
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._inventory = int(self._rng.integers(0, self.max_inventory + 1))
        self._last_demand = self.base_demand * 0.5
        self._steps = 0
        return self._obs(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
        price = self.price_min + (a[0] + 1.0) / 2.0 * (self.price_max - self.price_min)
        order = int(round((a[1] + 1.0) / 2.0 * self.max_order))

        inv_after = min(self._inventory + order, self.max_inventory)
        demand = int(self._rng.poisson(self._demand_mean(price)))
        sales = min(inv_after, demand)
        lost = demand - sales
        self._inventory = inv_after - sales
        self._last_demand = float(demand)

        reward = (price * sales - self.unit_cost * order
                  - self.holding_cost * self._inventory - self.stockout_penalty * lost)

        self._steps += 1
        truncated = self._steps >= self.horizon
        info = {"price": price, "order": order, "demand": demand, "sales": sales,
                "inventory": self._inventory}
        return self._obs(), float(reward), False, truncated, info

    def render_rgb(self):
        from ..utils.render import bars_frame
        return bars_frame(["inventory"], [self._inventory], self.max_inventory,
                          title="joint pricing + inventory")
