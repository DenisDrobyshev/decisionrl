import numpy as np
import pytest

from reinforce.algorithms import DDPG, SAC, TD3
from reinforce.envs import PointMass
from reinforce.training import evaluate_policy

ALGOS = [DDPG, TD3, SAC]


@pytest.mark.parametrize("cls", ALGOS)
def test_continuous_predict_within_bounds(cls, quiet_logger):
    env = PointMass()
    agent = cls(env, learning_starts=10, batch_size=8, seed=0, logger=quiet_logger)
    obs, _ = env.reset(seed=0)
    action = np.asarray(agent.predict(obs, deterministic=True))
    assert action.shape == (2,)
    assert np.all(action >= -1.0 - 1e-5) and np.all(action <= 1.0 + 1e-5)


@pytest.mark.parametrize("cls", ALGOS)
def test_continuous_prioritized_nstep_constructs(cls, quiet_logger):
    from reinforce.buffers import PrioritizedReplayBuffer

    agent = cls(PointMass(), prioritized=True, n_step=3, learning_starts=10, batch_size=8,
                seed=0, logger=quiet_logger)
    assert isinstance(agent.buffer, PrioritizedReplayBuffer)
    agent.learn(60)  # exercises the PER sampling + priority-update path
    assert agent.num_timesteps >= 60


@pytest.mark.slow
@pytest.mark.parametrize("cls,steps,threshold", [(DDPG, 3000, -15), (TD3, 3000, -15), (SAC, 3000, -30)])
def test_continuous_learns_pointmass(cls, steps, threshold, quiet_logger):
    agent = cls(PointMass(), learning_starts=200, batch_size=64, seed=0, logger=quiet_logger)
    agent.learn(steps)
    mean_return, _ = evaluate_policy(agent, PointMass(), n_episodes=20, seed=1)
    assert mean_return > threshold, f"{cls.__name__} failed on PointMass (mean_return={mean_return:.2f})"


@pytest.mark.slow
def test_sac_prioritized_nstep_learns(quiet_logger):
    agent = SAC(PointMass(), prioritized=True, n_step=3, learning_starts=200, batch_size=64,
                seed=0, logger=quiet_logger)
    agent.learn(3000)
    mean_return = evaluate_policy(agent, PointMass(), n_episodes=20, seed=1)[0]
    assert mean_return > -20, f"SAC+PER+n-step failed (mean_return={mean_return:.2f})"
