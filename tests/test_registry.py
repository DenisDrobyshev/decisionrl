import pytest

from reinforce import make_agent, make_env, make_vec_env
from reinforce.algorithms import PPO, QLearning
from reinforce.registry import list_algorithms, list_environments


def test_make_env_builtin():
    env = make_env("CartPole")
    assert env.observation_space.shape == (4,)


def test_make_agent_builtin():
    agent = make_agent("ppo", make_env("CartPole"), seed=0)
    assert isinstance(agent, PPO)
    agent2 = make_agent("QLearning", make_env("GridWorld"), seed=0)
    assert isinstance(agent2, QLearning)


def test_unknown_names_raise():
    with pytest.raises(KeyError):
        make_agent("nope", make_env("CartPole"))
    with pytest.raises(KeyError):
        make_env("NoSuchEnv")


def test_list_functions():
    assert "ppo" in list_algorithms()
    assert "CartPole" in list_environments()


def test_make_vec_env_sync():
    venv = make_vec_env("CartPole", n_envs=3)
    try:
        assert venv.num_envs == 3
        obs, _ = venv.reset(seed=0)
        assert obs.shape == (3, 4)
    finally:
        venv.close()


def test_make_vec_env_ppo_trains():
    venv = make_vec_env("CartPole", n_envs=2)
    try:
        agent = PPO(venv, n_steps=32, batch_size=16, n_epochs=1, seed=0)
        agent.learn(200)
        assert agent.num_timesteps >= 200
    finally:
        venv.close()


def test_make_env_gym_prefix():
    pytest.importorskip("gymnasium")
    env = make_env("gym:CartPole-v1")
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
