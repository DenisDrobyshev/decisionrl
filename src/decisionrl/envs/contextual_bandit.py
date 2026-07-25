"""A linear contextual bandit: pick one of K actions given a context vector.

Each round draws a context ``x`` (customer or market features). Every arm ``a`` has an
unknown linear value ``x . theta_a``; the reward is that value plus noise, and the best
arm depends on the context. This is the canonical testbed for contextual-bandit methods
(pricing, recommendation, allocation): the learner must generalise across contexts, not
just find one globally best arm.

One episode is a run of ``horizon`` rounds. The step ``info`` reports the per-round
regret (best expected reward minus the chosen arm's expected reward) and the optimal
arm, so a learner's cumulative regret can be measured exactly.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..core.env import Env
from ..core.spaces import Box, Discrete

__all__ = ["ContextualBandit"]


class ContextualBandit(Env):
    def __init__(
        self,
        n_arms: int = 6,
        n_features: int = 8,
        horizon: int = 2000,
        noise: float = 0.1,
        seed: Optional[int] = None,
    ) -> None:
        self.n_arms = int(n_arms)
        self.n_features = int(n_features)
        self.horizon = int(horizon)
        self.noise = float(noise)

        self.observation_space = Box(-np.inf, np.inf, shape=(self.n_features,), dtype=np.float32)
        self.action_space = Discrete(self.n_arms)

        # True per-arm parameters are fixed for the environment instance so that
        # different episodes/seeds share the same problem (only contexts and noise vary).
        param_rng = np.random.default_rng(0 if seed is None else seed)
        self.theta = param_rng.normal(size=(self.n_arms, self.n_features))
        self.theta /= np.linalg.norm(self.theta, axis=1, keepdims=True)

        self._rng = np.random.default_rng()
        self._context = np.zeros(self.n_features, dtype=np.float32)
        self._steps = 0

    def _new_context(self) -> np.ndarray:
        x = self._rng.normal(size=self.n_features)
        x /= np.linalg.norm(x) + 1e-8
        return x.astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._steps = 0
        self._context = self._new_context()
        return self._context.copy(), {}

    def step(self, action: int):
        arm = int(action)
        expected = self.theta @ self._context  # expected reward of every arm
        reward = float(expected[arm] + self.noise * self._rng.normal())
        best = int(np.argmax(expected))
        regret = float(expected[best] - expected[arm])

        self._steps += 1
        truncated = self._steps >= self.horizon
        info = {"regret": regret, "optimal_arm": best, "expected_reward": float(expected[arm])}
        self._context = self._new_context()
        return self._context.copy(), reward, False, truncated, info
