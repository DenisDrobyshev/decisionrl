import pytest

from reinforce.algorithms import RecurrentPPO
from reinforce.envs import CartPole
from reinforce.training import evaluate_policy
from reinforce.wrappers import SyncVectorEnv


def test_recurrent_ppo_predict_and_reset(quiet_logger):
    agent = RecurrentPPO(CartPole(), n_steps=16, n_epochs=1, seed=0, logger=quiet_logger)
    obs, _ = CartPole().reset(seed=0)
    agent.reset_states()
    assert agent.predict(obs, deterministic=True) in (0, 1)


def test_recurrent_ppo_vector_smoke(quiet_logger):
    venv = SyncVectorEnv([lambda: CartPole() for _ in range(3)])
    agent = RecurrentPPO(venv, n_steps=16, n_epochs=1, n_minibatches=1, seed=0, logger=quiet_logger)
    agent.learn(200)
    assert agent.num_timesteps >= 200


def test_recurrent_ppo_save_load(tmp_path, quiet_logger):
    agent = RecurrentPPO(CartPole(), n_steps=16, n_epochs=1, seed=0, logger=quiet_logger)
    agent.learn(64)
    path = str(tmp_path / "rppo.pt")
    agent.save(path)
    loaded = RecurrentPPO.load(path, env=CartPole())
    obs, _ = CartPole().reset(seed=1)
    agent.reset_states()
    loaded.reset_states()
    assert agent.predict(obs) == loaded.predict(obs)


@pytest.mark.slow
def test_recurrent_ppo_learns_cartpole(quiet_logger):
    venv = SyncVectorEnv([lambda: CartPole() for _ in range(4)])
    agent = RecurrentPPO(venv, n_steps=128, n_epochs=6, n_minibatches=4, ent_coef=0.0,
                         learning_rate=1e-3, seed=0, logger=quiet_logger)
    agent.learn(25_000)
    mean = evaluate_policy(agent, CartPole(), n_episodes=20)[0]
    assert mean > 60, f"RecurrentPPO failed to learn CartPole (mean_return={mean:.1f})"
