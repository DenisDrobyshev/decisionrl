"""Tests for the standalone container serving entrypoint (deploy/serve.py).

The entrypoint is validated here at the API level; that it runs without PyTorch or the
decisionrl package (the point of the small serving image) is proven separately by the
docker-serve job in CI, which runs it in an image that has neither installed.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy"
MODEL = DEPLOY / "models" / "policy.onnx"

pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="exported serving model not present")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("DECISIONRL_MODEL", str(MODEL))
    monkeypatch.syspath_prepend(str(DEPLOY))
    sys.modules.pop("serve", None)
    import serve
    from fastapi.testclient import TestClient

    return TestClient(serve.app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_info_reports_metadata(client):
    info = client.get("/info").json()
    assert info["obs_dim"] == 4
    assert info["action_type"] == "discrete"


def test_predict_returns_action(client):
    resp = client.post("/predict", json={"observation": [0.0, 0.0, 0.0, 0.0]})
    assert resp.status_code == 200
    assert "action" in resp.json()


def test_predict_rejects_wrong_observation_length(client):
    assert client.post("/predict", json={"observation": [0.0, 0.0]}).status_code == 422
