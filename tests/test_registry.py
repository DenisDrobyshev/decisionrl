import pytest

from reinforce import make_agent, make_env
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


def test_make_env_gym_prefix():
    pytest.importorskip("gymnasium")
    env = make_env("gym:CartPole-v1")
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
