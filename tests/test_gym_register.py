"""Gymnasium registration: decisionrl envs usable via gymnasium.make(...)."""

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from decisionrl.envs import (  # noqa: E402
    InventoryManagement,  # noqa: E402
    register_envs,
    to_gymnasium,
)


def test_register_envs_exposes_gymnasium_ids():
    ids = register_envs()
    all_ids = set(register_envs()) | set(ids)  # idempotent: second call adds nothing
    assert "decisionrl/InventoryManagement-v0" in all_ids
    assert register_envs() == []  # already registered


@pytest.mark.parametrize("env_id", ["decisionrl/InventoryManagement-v0",
                                    "decisionrl/EnergyMicrogrid-v0",
                                    "decisionrl/QueueAdmissionControl-v0"])
def test_gymnasium_make_roundtrip(env_id):
    register_envs()
    env = gym.make(env_id)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(np.asarray(obs, dtype=env.observation_space.dtype))
    obs, reward, term, trunc, info = env.step(env.action_space.sample())
    assert np.isfinite(reward)


def test_to_gymnasium_space_types():
    genv = to_gymnasium(InventoryManagement())
    assert isinstance(genv.observation_space, gym.spaces.Box)
    assert isinstance(genv.action_space, gym.spaces.Discrete)
