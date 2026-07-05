"""Observation transformation wrappers: frame stacking, flattening, one-hot."""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from ..core.env import Env, Wrapper
from ..core.spaces import Box, Dict, flatdim, flatten, is_discrete

__all__ = ["FrameStack", "FlattenObservation", "OneHotObservation", "FlattenDictObservation"]


class FrameStack(Wrapper):
    """Stack the last ``k`` observations along axis 0.

    For image observations ``(C, H, W)`` this yields ``(k * C, H, W)`` (the usual
    Atari trick to expose motion); for vector observations ``(n,)`` it yields
    ``(k * n,)``. Requires a ``Box`` observation space.
    """

    def __init__(self, env: Env, k: int = 4) -> None:
        super().__init__(env)
        assert isinstance(self.env.observation_space, Box), "FrameStack needs a Box space"
        self.k = int(k)
        self.frames: deque = deque(maxlen=self.k)
        low = np.concatenate([self.env.observation_space.low] * self.k, axis=0)
        high = np.concatenate([self.env.observation_space.high] * self.k, axis=0)
        self.observation_space = Box(low, high, dtype=self.env.observation_space.dtype)

    def _obs(self) -> np.ndarray:
        return np.concatenate(list(self.frames), axis=0)

    def reset(self, *, seed: Optional[int] = None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        obs = np.asarray(obs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(np.asarray(obs))
        return self._obs(), reward, terminated, truncated, info


class FlattenObservation(Wrapper):
    """Flatten a multi-dimensional Box observation to a 1-D vector."""

    def __init__(self, env: Env) -> None:
        super().__init__(env)
        space = self.env.observation_space
        assert isinstance(space, Box), "FlattenObservation needs a Box space"
        self.observation_space = Box(space.low.flatten(), space.high.flatten(), dtype=space.dtype)

    def reset(self, *, seed: Optional[int] = None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return np.asarray(obs).flatten(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return np.asarray(obs).flatten(), reward, terminated, truncated, info


class OneHotObservation(Wrapper):
    """Convert a Discrete observation into a one-hot Box vector.

    Lets tabular-style environments (integer states) be used with function
    approximation (DQN, PPO, ...).
    """

    def __init__(self, env: Env) -> None:
        super().__init__(env)
        assert is_discrete(self.env.observation_space), "OneHotObservation needs a Discrete space"
        self.n = int(self.env.observation_space.n)
        self.observation_space = Box(0.0, 1.0, shape=(self.n,), dtype=np.float32)

    def _encode(self, obs) -> np.ndarray:
        vec = np.zeros(self.n, dtype=np.float32)
        vec[int(obs)] = 1.0
        return vec

    def reset(self, *, seed: Optional[int] = None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._encode(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._encode(obs), reward, terminated, truncated, info


class FlattenDictObservation(Wrapper):
    """Flatten a :class:`~reinforce.core.spaces.Dict` observation into one Box.

    Each subspace is flattened (Discrete -> one-hot, Box -> raveled) and the
    parts are concatenated, so multi-modal environments work with the standard
    MLP-based agents.
    """

    def __init__(self, env: Env) -> None:
        super().__init__(env)
        space = self.env.observation_space
        assert isinstance(space, Dict), "FlattenDictObservation needs a Dict observation space"
        self._dict_space = space
        dim = flatdim(space)
        self.observation_space = Box(-np.inf, np.inf, shape=(dim,), dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return flatten(self._dict_space, obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return flatten(self._dict_space, obs), reward, terminated, truncated, info
