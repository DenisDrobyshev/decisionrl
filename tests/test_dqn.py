import numpy as np
import pytest

from reinforce.algorithms import C51, DQN
from reinforce.envs import GridWorld
from reinforce.training import evaluate_policy


def make_env(**kw):
    return GridWorld(rows=4, cols=4, one_hot=True, **kw)


@pytest.mark.parametrize(
    "opts",
    [
        dict(),
        dict(double_q=True),
        dict(dueling=True),
        dict(prioritized=True),
        dict(double_q=True, dueling=True, prioritized=True),
    ],
)
def test_dqn_constructs_and_predicts(opts, quiet_logger):
    agent = DQN(make_env(), learning_starts=10, batch_size=8, buffer_size=1000,
                seed=0, logger=quiet_logger, **opts)
    obs = np.eye(16, dtype=np.float32)[0]
    action = agent.predict(obs)
    assert 0 <= action < 4


@pytest.mark.slow
def test_dqn_learns_gridworld(quiet_logger):
    agent = DQN(
        make_env(), learning_rate=1e-3, learning_starts=500, batch_size=64,
        buffer_size=10_000, hidden_sizes=(64, 64), target_update_interval=200,
        seed=0, logger=quiet_logger,
    )
    agent.learn(8_000)
    mean_return, _ = evaluate_policy(agent, make_env(), n_episodes=10)
    assert mean_return > 0.5, f"DQN failed to learn GridWorld (mean_return={mean_return:.3f})"


def test_c51_constructs_and_predicts(quiet_logger):
    agent = C51(make_env(), v_min=-1.0, v_max=1.0, n_atoms=21, learning_starts=10,
                batch_size=8, buffer_size=1000, seed=0, logger=quiet_logger)
    action = agent.predict(np.eye(16, dtype=np.float32)[0])
    assert 0 <= action < 4


def test_dqn_nstep_constructs(quiet_logger):
    agent = DQN(make_env(), n_step=3, learning_starts=10, batch_size=8, buffer_size=1000,
                seed=0, logger=quiet_logger)
    agent.learn(200)
    assert agent.num_timesteps >= 200


@pytest.mark.slow
def test_c51_learns_gridworld(quiet_logger):
    agent = C51(make_env(), v_min=-1.0, v_max=1.0, n_atoms=51, learning_rate=1e-3,
                learning_starts=500, batch_size=64, buffer_size=10_000, hidden_sizes=(64, 64),
                target_update_interval=200, seed=0, logger=quiet_logger)
    agent.learn(8_000)
    mean_return, _ = evaluate_policy(agent, make_env(), n_episodes=10)
    assert mean_return > 0.5, f"C51 failed to learn GridWorld (mean_return={mean_return:.3f})"


@pytest.mark.slow
def test_dqn_prioritized_learns(quiet_logger):
    agent = DQN(
        make_env(), learning_rate=1e-3, learning_starts=500, batch_size=64,
        buffer_size=10_000, hidden_sizes=(64, 64), target_update_interval=200,
        double_q=True, prioritized=True, seed=0, logger=quiet_logger,
    )
    agent.learn(8_000)
    mean_return, _ = evaluate_policy(agent, make_env(), n_episodes=10)
    assert mean_return > 0.5, f"Prioritized DQN failed to learn (mean_return={mean_return:.3f})"
