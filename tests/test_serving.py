"""Tests for policy export (ONNX / TorchScript) and the FastAPI server."""

import numpy as np
import pytest

from reinforce.algorithms import PPO, SAC, QLearning
from reinforce.envs import CartPole, GridWorld, PointMass
from reinforce.serving import OnnxPolicy, build_policy_module, export_onnx, export_torchscript


def test_onnx_export_matches_discrete_agent(tmp_path, quiet_logger):
    agent = PPO(CartPole(), n_steps=64, n_epochs=1, seed=0, logger=quiet_logger)
    agent.learn(128)
    policy = OnnxPolicy(export_onnx(agent, str(tmp_path / "ppo.onnx")))
    assert policy.discrete
    for s in range(20):
        obs, _ = CartPole().reset(seed=s)
        assert policy.predict(obs) == agent.predict(obs, deterministic=True)


def test_onnx_export_matches_continuous_agent(tmp_path, quiet_logger):
    agent = SAC(PointMass(), learning_starts=50, batch_size=16, seed=0, logger=quiet_logger)
    agent.learn(150)
    policy = OnnxPolicy(export_onnx(agent, str(tmp_path / "sac.onnx")))
    assert not policy.discrete
    for s in range(10):
        obs, _ = PointMass().reset(seed=s)
        np.testing.assert_allclose(
            policy.predict(obs), np.asarray(agent.predict(obs, deterministic=True)), atol=1e-4
        )


def test_torchscript_export_loads_and_runs(tmp_path, quiet_logger):
    import torch

    agent = SAC(PointMass(), learning_starts=50, batch_size=16, seed=0, logger=quiet_logger)
    agent.learn(120)
    path = export_torchscript(agent, str(tmp_path / "sac.ts"))
    module = torch.jit.load(path)
    action = module(torch.zeros(1, 2)).detach().numpy()
    assert action.shape == (1, 2) and np.all(np.isfinite(action))


def test_export_rejects_unsupported_agent():
    agent = QLearning(GridWorld(rows=3, cols=3), seed=0)
    with pytest.raises(TypeError):
        build_policy_module(agent)


def test_fastapi_server_serves_predictions(tmp_path, quiet_logger):
    from fastapi.testclient import TestClient

    from reinforce.serving import create_app

    agent = PPO(CartPole(), n_steps=64, n_epochs=1, seed=0, logger=quiet_logger)
    agent.learn(128)
    model_path = export_onnx(agent, str(tmp_path / "ppo.onnx"))
    client = TestClient(create_app(model_path))

    assert client.get("/health").json() == {"status": "ok"}
    info = client.get("/info").json()
    assert info["obs_dim"] == 4 and info["action_type"] == "discrete"

    obs, _ = CartPole().reset(seed=0)
    resp = client.post("/predict", json={"observation": obs.tolist()})
    assert resp.status_code == 200
    assert resp.json()["action"] in (0, 1)

    bad = client.post("/predict", json={"observation": [0.0, 0.0]})  # wrong length
    assert bad.status_code == 422
