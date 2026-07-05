"""Multi-armed bandit: the simplest RL problem (a one-state MDP)."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from ..core.env import Env
from ..core.spaces import Box, Discrete

__all__ = ["MultiArmedBandit"]


class MultiArmedBandit(Env):
    """A stationary Gaussian multi-armed bandit.

    Each of ``n_arms`` arms yields a reward drawn from ``N(mean_i, sigma^2)``.
    The observation is constant (a single dummy state) since bandits are
    stateless; every episode is a single step. Great for validating exploration
    and value estimation in isolation.
    """

    def __init__(
        self,
        n_arms: int = 10,
        means: Optional[Sequence[float]] = None,
        sigma: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        self.n_arms = int(n_arms)
        self.sigma = float(sigma)
        self._rng = np.random.default_rng(seed)
        if means is None:
            self.means = self._rng.normal(0.0, 1.0, size=self.n_arms)
        else:
            self.means = np.asarray(means, dtype=np.float64)
            self.n_arms = len(self.means)

        self.action_space = Discrete(self.n_arms)
        self.observation_space = Box(0.0, 1.0, shape=(1,), dtype=np.float32)

    @property
    def optimal_arm(self) -> int:
        return int(np.argmax(self.means))

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action: int):
        action = int(action)
        assert self.action_space.contains(action), f"invalid action {action}"
        reward = float(self._rng.normal(self.means[action], self.sigma))
        obs = np.zeros(1, dtype=np.float32)
        # Bandits are single-step episodes: terminate immediately.
        info = {"optimal_arm": self.optimal_arm, "is_optimal": action == self.optimal_arm}
        return obs, reward, True, False, info
