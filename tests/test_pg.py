import numpy as np
import pytest

from reinforce.algorithms import A2C, PPO, REINFORCE
from reinforce.envs import CartPole, PointMass
from reinforce.training import evaluate_policy
from reinforce.wrappers import SyncVectorEnv


def cartpole():
    return CartPole(max_steps=500)


@pytest.mark.parametrize("cls", [PPO, A2C, REINFORCE])
def test_pg_discrete_predict(cls, quiet_logger):
    agent = cls(cartpole(), seed=0, logger=quiet_logger)
    obs, _ = cartpole().reset(seed=0)
    action = agent.predict(obs, deterministic=True)
    assert action in (0, 1)


@pytest.mark.parametrize("cls", [PPO, A2C, REINFORCE])
def test_pg_continuous_predict_within_bounds(cls, quiet_logger):
    env = PointMass()
    agent = cls(env, seed=0, logger=quiet_logger)
    obs, _ = env.reset(seed=0)
    action = agent.predict(obs, deterministic=True)
    action = np.asarray(action)
    assert action.shape == (2,)
    assert np.all(action >= -1.0 - 1e-5) and np.all(action <= 1.0 + 1e-5)


def test_ppo_anneal_lr_decays(quiet_logger):
    agent = PPO(cartpole(), n_steps=64, batch_size=32, n_epochs=1, learning_rate=1e-3,
                anneal_lr=True, seed=0, logger=quiet_logger)
    agent.learn(256)
    final_lr = agent.optimizer.param_groups[0]["lr"]
    assert final_lr < 1e-3  # learning rate was annealed downward


def test_ppo_accepts_vector_env(quiet_logger):
    venv = SyncVectorEnv([lambda: cartpole() for _ in range(3)])
    agent = PPO(venv, n_steps=32, batch_size=16, n_epochs=2, seed=0, logger=quiet_logger)
    agent.learn(200)
    assert agent.num_timesteps >= 200


@pytest.mark.slow
def test_ppo_learns_cartpole(quiet_logger):
    agent = PPO(cartpole(), n_steps=1024, batch_size=64, n_epochs=10, seed=0, logger=quiet_logger)
    agent.learn(20_000)
    mean_return, _ = evaluate_policy(agent, cartpole(), n_episodes=20)
    assert mean_return > 100, f"PPO failed to learn CartPole (mean_return={mean_return:.1f})"


@pytest.mark.slow
def test_a2c_learns_cartpole(quiet_logger):
    agent = A2C(cartpole(), n_steps=16, seed=0, logger=quiet_logger)
    agent.learn(25_000)
    mean_return, _ = evaluate_policy(agent, cartpole(), n_episodes=20)
    assert mean_return > 100, f"A2C failed to learn CartPole (mean_return={mean_return:.1f})"


@pytest.mark.slow
def test_reinforce_learns_cartpole(quiet_logger):
    agent = REINFORCE(cartpole(), learning_rate=1e-3, seed=0, logger=quiet_logger)
    agent.learn(25_000)
    mean_return, _ = evaluate_policy(agent, cartpole(), n_episodes=20)
    assert mean_return > 120, f"REINFORCE failed to learn CartPole (mean_return={mean_return:.1f})"
