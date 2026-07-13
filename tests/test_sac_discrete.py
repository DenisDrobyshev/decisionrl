import pytest

from decisionrl.algorithms import SACDiscrete
from decisionrl.envs import CartPole
from decisionrl.training import evaluate_policy


def test_sac_discrete_predict(quiet_logger):
    agent = SACDiscrete(CartPole(), learning_starts=10, batch_size=8, buffer_size=500,
                        seed=0, logger=quiet_logger)
    obs, _ = CartPole().reset(seed=0)
    assert agent.predict(obs, deterministic=True) in (0, 1)


def test_sac_discrete_save_load(tmp_path, quiet_logger):
    agent = SACDiscrete(CartPole(), learning_starts=10, batch_size=8, buffer_size=500,
                        seed=0, logger=quiet_logger)
    agent.learn(100)
    path = str(tmp_path / "sacd.pt")
    agent.save(path)
    loaded = SACDiscrete.load(path, env=CartPole())
    obs, _ = CartPole().reset(seed=1)
    assert agent.predict(obs) == loaded.predict(obs)


@pytest.mark.slow
def test_sac_discrete_learns_cartpole(quiet_logger):
    agent = SACDiscrete(CartPole(), learning_rate=3e-4, learning_starts=1000, batch_size=64,
                        buffer_size=50_000, hidden_sizes=(128, 128), tau=0.01, seed=0, logger=quiet_logger)
    agent.learn(15_000)
    mean_return, _ = evaluate_policy(agent, CartPole(), n_episodes=20)
    assert mean_return > 100, f"Discrete SAC failed to learn CartPole (mean_return={mean_return:.1f})"
