import numpy as np
import pytest

from reinforce.algorithms import IMPALA
from reinforce.envs import CartPole, PointMass
from reinforce.training import evaluate_policy
from reinforce.wrappers import SyncVectorEnv


def test_impala_discrete_predict(quiet_logger):
    agent = IMPALA(CartPole(), n_steps=16, n_epochs=1, seed=0, logger=quiet_logger)
    obs, _ = CartPole().reset(seed=0)
    assert agent.predict(obs, deterministic=True) in (0, 1)


def test_impala_continuous_predict(quiet_logger):
    agent = IMPALA(PointMass(), n_steps=16, n_epochs=1, seed=0, logger=quiet_logger)
    obs, _ = PointMass().reset(seed=0)
    action = np.asarray(agent.predict(obs, deterministic=True))
    assert action.shape == (2,) and np.all(np.abs(action) <= 1.0 + 1e-5)


def test_impala_vtrace_multi_epoch_smoke(quiet_logger):
    # multiple epochs make behaviour/target policies diverge -> exercises V-trace
    venv = SyncVectorEnv([lambda: CartPole() for _ in range(3)])
    agent = IMPALA(venv, n_steps=16, n_epochs=3, seed=0, logger=quiet_logger)
    agent.learn(200)
    assert agent.num_timesteps >= 200


@pytest.mark.slow
def test_impala_learns_cartpole(quiet_logger):
    venv = SyncVectorEnv([lambda: CartPole() for _ in range(8)])
    agent = IMPALA(venv, n_steps=32, n_epochs=4, ent_coef=0.01, learning_rate=3e-4,
                   seed=0, logger=quiet_logger)
    agent.learn(40_000)
    mean = evaluate_policy(agent, CartPole(), n_episodes=20)[0]
    assert mean > 150, f"IMPALA failed to learn CartPole (mean_return={mean:.1f})"
