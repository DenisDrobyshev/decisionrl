"""Inventory control driven by an empirical (real-world) demand series.

Unlike :class:`NonstationaryInventory`, whose demand comes from a two-regime Poisson
generator, this environment replays a demand series you supply, so it can be driven by
a real dataset. Each episode starts at a random offset into the series and steps
forward, drawing Poisson arrivals around the empirical level at each point. When the
series trends or shifts across eras (as real demand usually does), no single order-up-to
level is right everywhere, so a policy that reads recent demand can adapt and beat the
best fixed base-stock. Provide any 1-D array of non-negative demand levels:

    env = DatasetDemandInventory(demand_series=my_series)

See ``examples/real_data_case_study.py`` for an end-to-end study on real US consumption.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from ..core.env import Env
from ..core.spaces import Box, Discrete

__all__ = ["DatasetDemandInventory"]


class DatasetDemandInventory(Env):
    def __init__(
        self,
        demand_series: Sequence[float],
        max_inventory: int = 30,
        max_order: int = 18,
        demand_low: float = 3.0,
        demand_high: float = 15.0,
        price: float = 1.0,
        unit_cost: float = 0.3,
        holding_cost: float = 0.25,
        stockout_penalty: float = 0.8,
        horizon: int = 40,
    ) -> None:
        raw = np.asarray(demand_series, dtype=np.float64).reshape(-1)
        if raw.size < 2:
            raise ValueError("demand_series must contain at least two points")
        # Min-max rescale the empirical series onto [demand_low, demand_high] so its real
        # shape (trend, cycles, shocks) is preserved at a scale the inventory can serve.
        lo, hi = float(raw.min()), float(raw.max())
        span = hi - lo if hi > lo else 1.0
        self.demand = demand_low + (demand_high - demand_low) * (raw - lo) / span

        self.max_inventory = int(max_inventory)
        self.max_order = int(max_order)
        self.demand_low = float(demand_low)
        self.demand_high = float(demand_high)
        self.price = float(price)
        self.unit_cost = float(unit_cost)
        self.holding_cost = float(holding_cost)
        self.stockout_penalty = float(stockout_penalty)
        self.horizon = int(horizon)

        # Observation: inventory level and a smoothed (EWMA) read on recent demand,
        # both scaled to [0, 1]. The EWMA reveals the current demand era, which is what
        # lets an adaptive policy track it; no fixed order-up-to level fits every era.
        self.observation_space = Box(0.0, 1.0, shape=(2,), dtype=np.float32)
        self.action_space = Discrete(self.max_order + 1)  # order 0..max_order

        self._rng = np.random.default_rng()
        self._inventory = 0
        self._ewma = 0.0
        self._idx = 0
        self._steps = 0

    def _obs(self) -> np.ndarray:
        return np.array(
            [self._inventory / self.max_inventory,
             min(self._ewma / self.max_order, 1.0)],
            dtype=np.float32,
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._idx = int(self._rng.integers(0, len(self.demand)))
        self._inventory = int(self._rng.integers(0, self.max_inventory + 1))
        self._ewma = float(self.demand[self._idx])
        self._steps = 0
        return self._obs(), {}

    def step(self, action: int):
        order = int(np.clip(action, 0, self.max_order))
        inv_after_order = min(self._inventory + order, self.max_inventory)

        rate = float(self.demand[self._idx])
        demand = int(self._rng.poisson(rate))
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

        self._idx = (self._idx + 1) % len(self.demand)  # walk the series, wrap at the end
        self._steps += 1
        truncated = self._steps >= self.horizon
        info = {"demand": demand, "sales": sales, "lost_sales": lost, "order": order,
                "rate": rate}
        return self._obs(), float(reward), False, truncated, info

    def render_rgb(self):
        from ..utils.render import bars_frame
        return bars_frame(["inventory"], [self._inventory], self.max_inventory,
                          colors=["#2563eb"], title="dataset-driven demand")
