from typing import Optional

import numpy as np

from decisionrl.algorithms import PPO
from decisionrl.core.env import Env
from decisionrl.core.spaces import Box, Dict, Discrete, flatdim, flatten
from decisionrl.wrappers import FlattenDictObservation


class DictEnv(Env):
    """Tiny env with a Dict observation {pos: Box(2), id: Discrete(3)}."""

    def __init__(self, max_steps: int = 20) -> None:
        self.observation_space = Dict({"pos": Box(-1.0, 1.0, shape=(2,)), "id": Discrete(3)})
        self.action_space = Discrete(2)
        self.max_steps = max_steps
        self._rng = np.random.default_rng()
        self._steps = 0

    def _obs(self):
        return {"pos": self._rng.uniform(-1, 1, size=2).astype(np.float32), "id": int(self._rng.integers(3))}

    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._steps = 0
        return self._obs(), {}

    def step(self, action):
        self._steps += 1
        truncated = self._steps >= self.max_steps
        return self._obs(), 1.0, False, truncated, {}


def test_dict_space_flatdim_and_flatten():
    space = Dict({"a": Box(-1.0, 1.0, shape=(3,)), "b": Discrete(4)})
    assert flatdim(space) == 3 + 4
    sample = {"a": np.zeros(3, np.float32), "b": 2}
    flat = flatten(space, sample)
    assert flat.shape == (7,)
    assert flat[3 + 2] == 1.0  # one-hot component for b == 2
    assert space.contains(space.sample())


def test_flatten_dict_observation_shape():
    env = FlattenDictObservation(DictEnv())
    assert env.observation_space.shape == (2 + 3,)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (5,)
    obs, r, term, trunc, _ = env.step(env.action_space.sample())
    assert obs.shape == (5,)


def test_ppo_trains_on_dict_env(quiet_logger):
    env = FlattenDictObservation(DictEnv())
    agent = PPO(env, n_steps=32, batch_size=16, n_epochs=1, seed=0, logger=quiet_logger)
    agent.learn(100)
    assert agent.num_timesteps >= 100
