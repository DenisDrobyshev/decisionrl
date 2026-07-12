"""Tests for the Diffusion Policy (conditional denoising diffusion over actions)."""

import numpy as np
import pytest

from reinforce.algorithms import DiffusionPolicy
from reinforce.data import collect_dataset
from reinforce.envs import PointMass
from reinforce.training import evaluate_policy


def _expert(o):
    return np.clip(-np.asarray(o), -1.0, 1.0)  # move toward the origin


def test_diffusion_policy_predicts_and_round_trips(tmp_path, quiet_logger):
    data = collect_dataset(PointMass(), _expert, 500, seed=0)
    dp = DiffusionPolicy(PointMass(), n_diffusion_steps=10, seed=0, logger=quiet_logger)
    dp.train(data, n_iters=50, batch_size=64)

    obs, _ = PointMass().reset(seed=1)
    action = np.asarray(dp.predict(obs, deterministic=True))
    assert action.shape == (2,)
    assert np.all(action >= PointMass().action_space.low - 1e-6)
    assert np.all(action <= PointMass().action_space.high + 1e-6)

    path = str(tmp_path / "dp.pt")
    dp.save(path)
    loaded = DiffusionPolicy.load(path, env=PointMass())
    np.testing.assert_allclose(
        dp.predict(obs, deterministic=True), loaded.predict(obs, deterministic=True), atol=1e-5
    )


@pytest.mark.slow
def test_diffusion_policy_imitates_expert(quiet_logger):
    data = collect_dataset(PointMass(), _expert, 4000, seed=0)
    dp = DiffusionPolicy(PointMass(), n_diffusion_steps=20, seed=0, logger=quiet_logger)
    dp.train(data, n_iters=3000, batch_size=128)
    mean_return, _ = evaluate_policy(dp, PointMass(), n_episodes=15, seed=100)
    assert mean_return > -12.0  # expert is ~ -7.2; random is far worse
