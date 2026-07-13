"""Dynamic portfolio allocation — a financial sequential-decision task.

Allocate capital across ``n_assets`` risky assets whose returns follow a
correlated AR(1) *momentum* process: an asset that rose recently is more likely
to keep rising, so recent returns (part of the observation) are genuinely
predictive and the optimal policy is *not* static. Each rebalance pays a
proportional transaction cost, so the agent must trade off chasing momentum
against churn. The reward is the realized log-return net of costs.

Action = desired portfolio weights (mapped to the simplex via softmax). A learned
policy should beat naive equal-weight / buy-and-hold baselines. Solve with SAC or
PPO (continuous).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..core.env import Env
from ..core.spaces import Box

__all__ = ["PortfolioAllocation"]


class PortfolioAllocation(Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        n_assets: int = 4,
        horizon: int = 120,
        transaction_cost: float = 0.002,
        momentum: float = 0.3,
        vol: float = 0.02,
        drift_spread: float = 0.0004,
        market_seed: int = 0,
    ) -> None:
        self.n = int(n_assets)
        self.horizon = int(horizon)
        self.cost = float(transaction_cost)
        self.phi = float(momentum)
        self.vol = float(vol)

        # Fixed "market": per-asset baseline drift and a correlated noise structure,
        # deterministic across episodes so the task is stationary and learnable.
        market_rng = np.random.default_rng(market_seed)
        self.mu = market_rng.uniform(-drift_spread, drift_spread, size=self.n).astype(np.float64)
        a = market_rng.normal(size=(self.n, self.n)) * 0.3
        cov = a @ a.T / self.n + np.eye(self.n) * 1.0
        self.chol = np.linalg.cholesky(cov).astype(np.float64)

        # obs: current weights (n), last returns (n), rolling mean (n), rolling vol (n)
        self.observation_space = Box(-np.inf, np.inf, shape=(4 * self.n,), dtype=np.float32)
        self.action_space = Box(-1.0, 1.0, shape=(self.n,), dtype=np.float32)

        self._rng = np.random.default_rng()
        self._reset_state()

    def _reset_state(self) -> None:
        self._weights = np.ones(self.n) / self.n
        self._last_ret = np.zeros(self.n)
        self._ret_hist = [np.zeros(self.n)]
        self._step = 0

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        z = x - np.max(x)
        e = np.exp(z)
        return e / np.sum(e)

    def _obs(self) -> np.ndarray:
        hist = np.asarray(self._ret_hist[-20:])
        mean = hist.mean(axis=0)
        vol = hist.std(axis=0)
        return np.concatenate([self._weights, self._last_ret, mean, vol]).astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._reset_state()
        return self._obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(self.n)
        target = self._softmax(action)  # desired weights on the simplex

        # Transaction cost on turnover from current to target weights.
        turnover = float(np.sum(np.abs(target - self._weights)))
        cost = self.cost * turnover

        # AR(1) momentum returns with correlated shocks.
        shock = self.chol @ self._rng.normal(size=self.n) * self.vol
        returns = self.mu + self.phi * self._last_ret + shock
        self._last_ret = returns
        self._ret_hist.append(returns)

        port_return = float(np.dot(target, returns))
        net = port_return - cost
        reward = float(np.log1p(net))

        # Weights drift with realized returns until the next rebalance.
        grown = target * (1.0 + returns)
        self._weights = grown / np.sum(grown)

        self._step += 1
        truncated = self._step >= self.horizon
        info = {"port_return": port_return, "cost": cost, "turnover": turnover}
        return self._obs(), reward, False, truncated, info
