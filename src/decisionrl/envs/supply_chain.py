"""Two-echelon supply chain: an applied-RL operations problem (the "beer game").

A retailer faces stochastic customer demand and is replenished by a warehouse,
which is itself replenished by an unlimited supplier; every shipment takes one
period to arrive. Each step the agent sets both order quantities. It must hold
enough stock to avoid stockouts (penalized) without piling up holding cost at
either echelon, and coordinate the two so orders don't oscillate (the "bullwhip"
effect). A per-echelon base-stock rule is the classic baseline to beat.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..core.env import Env
from ..core.spaces import Box

__all__ = ["SupplyChain"]


class SupplyChain(Env):
    def __init__(
        self,
        demand_mean: float = 5.0,
        max_order: float = 15.0,
        holding_cost_retail: float = 0.1,
        holding_cost_warehouse: float = 0.05,
        stockout_penalty: float = 0.5,
        horizon: int = 60,
    ) -> None:
        self.demand_mean = float(demand_mean)
        self.max_order = float(max_order)
        self.h_ret = float(holding_cost_retail)
        self.h_wh = float(holding_cost_warehouse)
        self.stockout_penalty = float(stockout_penalty)
        self.horizon = int(horizon)
        self._scale = 40.0  # normalization constant for observations

        # Observation: retail & warehouse on-hand, both in-transit pipelines, last demand.
        self.observation_space = Box(0.0, 1.0, shape=(5,), dtype=np.float32)
        # Action: order-up amounts for [retailer, warehouse], each in [0, 1] of max_order.
        self.action_space = Box(0.0, 1.0, shape=(2,), dtype=np.float32)

        self._rng = np.random.default_rng()
        self._reset_state()

    def _reset_state(self) -> None:
        self._retail_inv = self.demand_mean
        self._wh_inv = self.demand_mean
        self._retail_pipe = self.demand_mean
        self._wh_pipe = self.demand_mean
        self._last_demand = self.demand_mean
        self._steps = 0

    def _obs(self) -> np.ndarray:
        return np.clip(
            np.array(
                [self._retail_inv, self._wh_inv, self._retail_pipe, self._wh_pipe, self._last_demand],
                dtype=np.float32,
            )
            / self._scale,
            0.0,
            1.0,
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._reset_state()
        return self._obs(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), 0.0, 1.0)
        order_ret = float(a[0]) * self.max_order
        order_wh = float(a[1]) * self.max_order

        # 1. Receive in-transit shipments from the previous period.
        self._retail_inv += self._retail_pipe
        self._wh_inv += self._wh_pipe

        # 2. Warehouse ships to the retailer (limited by its stock); supplier is unlimited.
        shipped = min(order_ret, self._wh_inv)
        self._wh_inv -= shipped
        self._retail_pipe = shipped
        self._wh_pipe = order_wh

        # 3. Customer demand hits the retailer (lost sales if short).
        demand = int(self._rng.poisson(self.demand_mean))
        sales = min(self._retail_inv, demand)
        unmet = demand - sales
        self._retail_inv -= sales
        self._last_demand = float(demand)

        # 4. Costs.
        holding = self.h_ret * self._retail_inv + self.h_wh * self._wh_inv
        reward = -(holding + self.stockout_penalty * unmet)

        self._steps += 1
        truncated = self._steps >= self.horizon
        info = {
            "demand": demand,
            "unmet": unmet,
            "retail_inv": self._retail_inv,
            "warehouse_inv": self._wh_inv,
        }
        return self._obs(), float(reward), False, truncated, info

    def render_rgb(self):
        from ..utils.render import bars_frame
        return bars_frame(["retailer", "warehouse"], [self._retail_inv, self._wh_inv],
                          self._scale, colors=["#2563eb", "#f59e0b"], title="supply chain")
