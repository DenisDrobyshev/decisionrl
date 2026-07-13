"""Truncate episodes after a fixed number of steps."""

from __future__ import annotations

from typing import Dict, Optional

from ..core.env import Env, Wrapper

__all__ = ["TimeLimit"]


class TimeLimit(Wrapper):
    """Emit ``truncated=True`` once ``max_episode_steps`` is reached.

    Correctly sets ``truncated`` (not ``terminated``) so downstream algorithms
    still bootstrap from the final observation, which is the whole point of the
    Gymnasium terminated/truncated split.
    """

    def __init__(self, env: Env, max_episode_steps: int) -> None:
        super().__init__(env)
        self.max_episode_steps = int(max_episode_steps)
        self._elapsed = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        self._elapsed = 0
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed += 1
        if self._elapsed >= self.max_episode_steps and not terminated:
            truncated = True
        return obs, reward, terminated, truncated, info
