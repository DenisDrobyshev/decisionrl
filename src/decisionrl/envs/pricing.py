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
        self._history_inventory=[]
        self._history_price=[]

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
        self._history_inventory=[self._inventory]
        self._history_price=[]
        return self._obs(), {}

    def step(self, action: int):
        price = float(self.prices[int(action)])
        demand = int(self._rng.poisson(self._demand_mean(price)))
        sales = min(self._inventory, demand)
        self._inventory -= sales
        reward = price * sales

        self._steps += 1
        self._history_inventory.append(self._inventory)
        self._history_price.append(price)
        terminated = self._inventory <= 0
        truncated = self._steps >= self.horizon
        info = {"price": price, "demand": demand, "sales": sales, "inventory": self._inventory}
        return self._obs(), float(reward), terminated, truncated, info


    def render_rgb(self):
        # Force matplotlib to use a non-interactive backend so we don't
        # accidentally pop up GUI windows during headless training loops.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from ..utils.render import fig_to_rgb

        # Set up a small, low-res canvas specifically for GIF generation
        fig, ax1 = plt.subplots(figsize=(4.0, 3.2), dpi=64)

        # Plot the inventory draining over time on the primary left y-axis
        x_inv = np.arange(len(self._history_inventory))
        ax1.plot(x_inv, self._history_inventory, color="#2563eb", lw=4, solid_capstyle="round")
        ax1.set_xlim(0, self.horizon)
        ax1.set_ylim(0, self.initial_inventory * 1.05) # Add a tiny bit of visual headroom
        ax1.set_ylabel("Inventory", color="#2563eb", fontweight="bold")

        # If we've taken steps, overlay the price history on a secondary right y-axis
        if self._history_price:
            ax2 = ax1.twinx()
            x_price = np.arange(1, len(self._history_price) + 1)

            # Using a step plot here makes more sense since prices are discrete decisions
            ax2.step(x_price, self._history_price, color="#f59e0b", lw=3, where="pre")
            ax2.set_ylim(self.price_min * 0.9, self.price_max * 1.1)
            ax2.set_ylabel("Price", color="#f59e0b", fontweight="bold")
            ax2.spines["right"].set_color("#f59e0b")

        # Match spine colors to their respective lines for readability
        ax1.spines["left"].set_color("#2563eb")
        fig.tight_layout() # Prevent axis labels from getting chopped off in the final array

        # Rasterize the figure to a numpy array and explicitly close it to prevent memory leaks
        frame = fig_to_rgb(fig)
        plt.close(fig)
        return frame
