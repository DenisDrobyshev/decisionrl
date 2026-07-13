import numpy as np
import pytest

from decisionrl.algorithms import TRPO
from decisionrl.envs import CartPole, PointMass
from decisionrl.training import evaluate_policy


def cartpole():
    return CartPole(max_steps=500)


def test_trpo_discrete_predict(quiet_logger):
    agent = TRPO(cartpole(), seed=0, logger=quiet_logger)
    obs, _ = cartpole().reset(seed=0)
    action = agent.predict(obs, deterministic=True)
    assert action in (0, 1)


def test_trpo_continuous_predict_within_bounds(quiet_logger):
    env = PointMass()
    agent = TRPO(env, seed=0, logger=quiet_logger)
    obs, _ = env.reset(seed=0)
    action = np.asarray(agent.predict(obs, deterministic=True))
    assert action.shape == (2,)
    assert np.all(action >= -1.0 - 1e-5) and np.all(action <= 1.0 + 1e-5)


def test_trpo_update_respects_trust_region(quiet_logger):
    # After an update the realized KL must stay within (a small slack over) max_kl.
    agent = TRPO(cartpole(), n_steps=256, max_kl=0.01, seed=0, logger=quiet_logger)
    agent.learn(256)
    info = agent._update()
    assert info["kl"] <= 0.01 + 1e-3
    assert 0.0 <= info["step_frac"] <= 1.0


def test_trpo_save_load_roundtrip(tmp_path, quiet_logger):
    agent = TRPO(cartpole(), n_steps=128, seed=0, logger=quiet_logger)
    agent.learn(128)
    obs, _ = cartpole().reset(seed=1)
    before = agent.predict(obs, deterministic=True)
    path = tmp_path / "trpo.pt"
    agent.save(path)
    restored = TRPO.load(path, env=cartpole())
    after = restored.predict(obs, deterministic=True)
    assert before == after


@pytest.mark.slow
def test_trpo_learns_cartpole(quiet_logger):
    agent = TRPO(cartpole(), n_steps=1024, seed=0, logger=quiet_logger)
    agent.learn(50_000)
    mean_return, _ = evaluate_policy(agent, cartpole(), n_episodes=20)
    assert mean_return > 150, f"TRPO failed to learn CartPole (mean_return={mean_return:.1f})"
