"""Tests for the model zoo (save / list / load pretrained policies)."""

import os
import shutil
import sys
import types

from decisionrl.algorithms import PPO
from decisionrl.envs import CartPole
from decisionrl.zoo import (
    list_pretrained,
    load_from_hub,
    load_pretrained,
    model_card,
    push_to_hub,
    save_to_zoo,
)


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


def test_model_card_has_frontmatter_and_usage():
    card = model_card("user/cartpole-ppo", "policy.onnx",
                      {"obs_dim": 4, "action_type": "discrete"})
    assert card.startswith("---\n")
    assert "library_name: decisionrl" in card
    assert "load_from_hub" in card
    assert "user/cartpole-ppo" in card
    assert "policy.onnx" in card
    assert "| Observation dimension | 4 |" in card


def _fake_hub(remote_dir):
    """A stand-in huggingface_hub module backed by a local directory (no network)."""
    class FakeApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, repo_id, **kw):
            pass

        def upload_folder(self, repo_id, folder_path, **kw):
            for name in os.listdir(folder_path):
                shutil.copy(os.path.join(folder_path, name), os.path.join(remote_dir, name))

    def hf_hub_download(repo_id, filename, **kw):
        return os.path.join(remote_dir, filename)

    mod = types.ModuleType("huggingface_hub")
    mod.HfApi = FakeApi
    mod.hf_hub_download = hf_hub_download
    return mod


def test_push_and_load_from_hub_roundtrip(tmp_path, quiet_logger, monkeypatch):
    remote = tmp_path / "remote"
    remote.mkdir()
    monkeypatch.setitem(sys.modules, "huggingface_hub", _fake_hub(str(remote)))

    agent = PPO(CartPole(), n_steps=64, n_epochs=1, seed=0, logger=quiet_logger)
    agent.learn(128)

    url = push_to_hub(agent, "user/cartpole-ppo")
    assert url == "https://huggingface.co/user/cartpole-ppo"
    assert (remote / "policy.onnx").exists()
    assert (remote / "policy.onnx.json").exists()
    assert (remote / "README.md").exists()

    policy = load_from_hub("user/cartpole-ppo")
    for s in range(5):
        obs, _ = CartPole().reset(seed=s)
        assert policy.predict(obs) == agent.predict(obs, deterministic=True)
