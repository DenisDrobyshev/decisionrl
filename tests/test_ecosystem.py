"""Tests for ecosystem adapters: MiniGrid and PettingZoo (skip if not installed)."""

import importlib.util

import numpy as np
import pytest

from decisionrl.envs import make_minigrid
from decisionrl.multiagent import PettingZooParallelAdapter, make_pettingzoo


def test_make_minigrid_error_or_runs():
    if importlib.util.find_spec("minigrid") is None:
        with pytest.raises(ImportError):
            make_minigrid("MiniGrid-Empty-5x5-v0")
        return
    env = make_minigrid("MiniGrid-Empty-5x5-v0")
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(np.asarray(obs, dtype=np.float32))


def test_make_pettingzoo_error_or_runs():
    if importlib.util.find_spec("pettingzoo") is None:
        with pytest.raises(ImportError):
            make_pettingzoo("pettingzoo.mpe.simple_spread_v3")
        return
    env = make_pettingzoo("pettingzoo.mpe.simple_spread_v3")
    assert isinstance(env, PettingZooParallelAdapter)
    obs, _ = env.reset(seed=0)
    assert set(obs) == set(env.agents)
