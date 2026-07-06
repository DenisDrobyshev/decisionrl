import numpy as np
import pytest

from reinforce.algorithms import MBPO
from reinforce.envs import PointMass
from reinforce.networks import EnsembleDynamics
from reinforce.training import evaluate_policy


def test_mbpo_constructs_and_rolls_out(quiet_logger):
    agent = MBPO(PointMass(), ensemble_size=3, learning_starts=10, batch_size=8,
                 model_train_freq=10, model_rollouts=20, seed=0, logger=quiet_logger)
    assert isinstance(agent.dynamics, EnsembleDynamics)
    agent.learn(60)  # triggers ensemble step + synthetic rollouts + mixed SAC update
    assert len(agent.model_buffer) > 0
    obs, _ = PointMass().reset(seed=0)
    assert np.asarray(agent.predict(obs, deterministic=True)).shape == (2,)


@pytest.mark.slow
def test_mbpo_learns_pointmass(quiet_logger):
    agent = MBPO(PointMass(), learning_starts=200, batch_size=64, real_ratio=0.5,
                 model_train_freq=250, model_rollouts=400, rollout_length=1,
                 seed=0, logger=quiet_logger)
    agent.learn(3000)
    mean = evaluate_policy(agent, PointMass(), n_episodes=20, seed=1)[0]
    assert mean > -20, f"MBPO failed to learn PointMass (mean_return={mean:.2f})"
