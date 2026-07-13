"""Tests for GRPO (Group Relative Policy Optimization)."""

import numpy as np
import pytest

from decisionrl.algorithms import GRPO
from decisionrl.envs import CartPole
from decisionrl.training import evaluate_policy


def test_grpo_predicts_valid_actions(quiet_logger):
    agent = GRPO(CartPole(), seed=0, logger=quiet_logger)
    obs, _ = CartPole().reset(seed=0)
    for deterministic in (True, False):
        action = agent.predict(obs, deterministic=deterministic)
        assert isinstance(action, (int, np.integer))
        assert CartPole().action_space.contains(int(action))


def test_grpo_save_load_predictions_match(tmp_path, quiet_logger):
    agent = GRPO(CartPole(), group_size=4, groups_per_update=2, seed=0, logger=quiet_logger)
    agent.learn(600)
    path = str(tmp_path / "grpo.pt")
    agent.save(path)

    loaded = GRPO.load(path, env=CartPole())
    obs, _ = CartPole().reset(seed=1)
    assert agent.predict(obs, deterministic=True) == loaded.predict(obs, deterministic=True)


@pytest.mark.slow
def test_grpo_learns_cartpole(quiet_logger):
    agent = GRPO(
        CartPole(), group_size=8, groups_per_update=4, n_epochs=4,
        learning_rate=1e-3, seed=0, logger=quiet_logger,
    )
    agent.learn(30_000)
    mean_return, _ = evaluate_policy(agent, CartPole(), n_episodes=10, seed=100)
    assert mean_return > 150.0
