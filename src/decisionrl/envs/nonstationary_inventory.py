"""Inventory with drifting (non-stationary) demand — where the classic formula breaks.

The base-stock ("order up to S") policy is *provably optimal* — but only when demand
is stationary. Here the demand rate follows a random walk over the episode, so no
single S is right for long: a level tuned for the average is too low in a surge and
too high in a lull. There is no clean closed-form order-up-to level for a drifting,
partially observed rate, which is exactly when reinforcement learning earns its keep.

The agent sees its inventory **and the most recent demand** (a noisy read on the
current rate), so it can *track* the drift and adapt its order-up-to level — beating
the best fixed base-stock. Compare against :class:`InventoryManagement` (stationary),
where RL only *matches* the base-stock optimum.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..core.env import Env
from ..core.spaces import Box, Discrete

__all__ = ["NonstationaryInventory"]


class NonstationaryInventory(Env):
    def __init__(
        self,
        max_inventory: int = 30,
        max_order: int = 18,
        demand_low: float = 2.0,
        demand_high: float = 14.0,
        switch_prob: float = 0.05,
        price: float = 1.0,
        unit_cost: float = 0.3,
        holding_cost: float = 0.25,
        stockout_penalty: float = 0.8,
        horizon: int = 80,
    ) -> None:
        self.max_inventory = int(max_inventory)
        self.max_order = int(max_order)
        self.demand_low = float(demand_low)
        self.demand_high = float(demand_high)
        self.switch_prob = float(switch_prob)
        self.price = float(price)
        self.unit_cost = float(unit_cost)
        self.holding_cost = float(holding_cost)
        self.stockout_penalty = float(stockout_penalty)
        self.horizon = int(horizon)

        # Observation: inventory level and a smoothed (EWMA) read on recent demand,
        # both scaled to [0, 1]. The EWMA is a reliable signal of the current regime,
        # which is what lets an adaptive policy track it — but no *fixed* order-up-to
        # level can be right in both the low and high regime at once.
        self.observation_space = Box(0.0, 1.0, shape=(2,), dtype=np.float32)
        self.action_space = Discrete(self.max_order + 1)  # order 0..max_order

        self._rng = np.random.default_rng()
        self._inventory = 0
        self._high = False
        self._ewma = 0.0
        self._steps = 0

    @property
    def _mu(self) -> float:
        return self.demand_high if self._high else self.demand_low

    def _obs(self) -> np.ndarray:
        return np.array(
            [self._inventory / self.max_inventory,
             min(self._ewma / self.max_order, 1.0)],
            dtype=np.float32,
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._inventory = int(self._rng.integers(0, self.max_inventory + 1))
        self._high = bool(self._rng.random() < 0.5)
        self._ewma = self._mu
        self._steps = 0
        return self._obs(), {}

    def step(self, action: int):
        order = int(np.clip(action, 0, self.max_order))
        inv_after_order = min(self._inventory + order, self.max_inventory)

        demand = int(self._rng.poisson(self._mu))
        sales = min(inv_after_order, demand)
        lost = demand - sales
        self._inventory = inv_after_order - sales
        self._ewma = 0.5 * self._ewma + 0.5 * demand

        reward = (
            self.price * sales
            - self.unit_cost * order
            - self.holding_cost * self._inventory
            - self.stockout_penalty * lost
        )

        # Persistent regime that occasionally flips — so recent demand is informative,
        # but a single fixed base-stock level is wrong for long stretches.
        if self._rng.random() < self.switch_prob:
            self._high = not self._high

        self._steps += 1
        truncated = self._steps >= self.horizon
        info = {"demand": demand, "sales": sales, "lost_sales": lost, "order": order,
                "regime": "high" if self._high else "low"}
        return self._obs(), float(reward), False, truncated, info

    def render_rgb(self):
        from ..utils.render import bars_frame
        color = "#ef4444" if self._high else "#2563eb"
        regime = "high" if self._high else "low"
        return bars_frame(["inventory"], [self._inventory], self.max_inventory,
                          colors=[color], title=f"demand regime: {regime}")
