"""Running normalization of observations and rewards.

These are among the most impactful "best practices" for on-policy algorithms
(PPO/A2C) and continuous control. Statistics are updated online with
:class:`~reinforce.utils.running_mean_std.RunningMeanStd`.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..core.env import Env, Wrapper
from ..core.spaces import Box
from ..utils.running_mean_std import RunningMeanStd

__all__ = ["NormalizeObservation", "NormalizeReward"]


class NormalizeObservation(Wrapper):
    """Standardize observations to zero mean / unit variance online."""

    def __init__(self, env: Env, epsilon: float = 1e-8, clip: float = 10.0) -> None:
        super().__init__(env)
        assert isinstance(self.observation_space, Box), "NormalizeObservation needs a Box space"
        self.rms = RunningMeanStd(shape=self.observation_space.shape)
        self.epsilon = float(epsilon)
        self.clip = float(clip)
        low = np.full(self.observation_space.shape, -clip, dtype=np.float32)
        high = np.full(self.observation_space.shape, clip, dtype=np.float32)
        self.observation_space = Box(low, high, dtype=np.float32)

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        self.rms.update(obs[None])
        out = (obs - self.rms.mean) / np.sqrt(self.rms.var + self.epsilon)
        return np.clip(out, -self.clip, self.clip).astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._normalize(np.asarray(obs, dtype=np.float32)), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._normalize(np.asarray(obs, dtype=np.float32)), reward, terminated, truncated, info


class NormalizeReward(Wrapper):
    """Scale rewards by the std of the discounted return estimate.

    Follows the widely-used implementation from the PPO/OpenAI-baselines lineage:
    rewards are divided by (but not centered on) the running standard deviation
    of the discounted returns.
    """

    def __init__(self, env: Env, gamma: float = 0.99, epsilon: float = 1e-8, clip: float = 10.0) -> None:
        super().__init__(env)
        self.rms = RunningMeanStd(shape=())
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.clip = float(clip)
        self._ret = 0.0

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        self._ret = 0.0
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._ret = self._ret * self.gamma + reward
        self.rms.update(np.array([self._ret]))
        norm_reward = reward / np.sqrt(self.rms.var + self.epsilon)
        norm_reward = float(np.clip(norm_reward, -self.clip, self.clip))
        if terminated or truncated:
            self._ret = 0.0
        return obs, norm_reward, terminated, truncated, info
