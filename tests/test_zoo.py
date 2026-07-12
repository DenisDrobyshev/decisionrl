"""Tests for the model zoo (save / list / load pretrained policies)."""


from reinforce.algorithms import PPO
from reinforce.envs import CartPole
from reinforce.zoo import list_pretrained, load_pretrained, save_to_zoo


def test_save_list_load_roundtrip(tmp_path, quiet_logger):
    zoo = str(tmp_path)
    agent = PPO(CartPole(), n_steps=64, n_epochs=1, seed=0, logger=quiet_logger)
    agent.learn(128)
    save_to_zoo(agent, "cartpole-ppo", zoo_dir=zoo)

    assert list_pretrained(zoo) == ["cartpole-ppo"]

    policy = load_pretrained("cartpole-ppo", zoo_dir=zoo)
    for s in range(10):
        obs, _ = CartPole().reset(seed=s)
        assert policy.predict(obs) == agent.predict(obs, deterministic=True)


def test_load_missing_raises(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        load_pretrained("nope", zoo_dir=str(tmp_path))
