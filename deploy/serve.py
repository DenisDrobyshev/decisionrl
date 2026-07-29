"""Standalone, dependency-light HTTP server for an exported ONNX policy.

This is the container entrypoint (see deploy/Dockerfile). It deliberately does NOT import
the ``decisionrl`` package: inference needs only onnxruntime, NumPy, and FastAPI, so the
serving image stays small and free of the PyTorch training stack. It mirrors the API of
``decisionrl.serving.server`` and reads the same artifact format that
``decisionrl.serving.export_onnx`` produces (``policy.onnx`` plus a ``policy.onnx.json``
metadata sidecar).

    DECISIONRL_MODEL=/models/policy.onnx uvicorn serve:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health   -> liveness probe
    GET  /info     -> policy metadata (obs dim, action type, bounds)
    POST /predict  -> {"observation": [...]} -> {"action": ...}
"""

from __future__ import annotations

import json
import os

import numpy as np
import onnxruntime as ort
from fastapi import Body, FastAPI, HTTPException


class OnnxPolicy:
    """Load an exported ONNX policy and run inference with onnxruntime + NumPy."""

    def __init__(self, model_path: str) -> None:
        with open(model_path + ".json", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    @property
    def discrete(self) -> bool:
        return self.meta["action_type"] == "discrete"

    def predict(self, obs):
        x = np.asarray(obs, dtype=np.float32).reshape(1, int(self.meta["obs_dim"]))
        out = self.session.run(None, {self.input_name: x})[0]
        if self.discrete:
            return int(np.asarray(out).reshape(-1)[0])
        return np.asarray(out, dtype=np.float32).reshape(-1)


def create_app(model_path: str) -> FastAPI:
    policy = OnnxPolicy(model_path)
    obs_dim = int(policy.meta["obs_dim"])
    app = FastAPI(title="decisionrl policy server", version="1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/info")
    def info() -> dict:
        return policy.meta

    @app.post("/predict")
    def predict(payload: dict = Body(...)) -> dict:
        obs = payload.get("observation")
        if not isinstance(obs, list):
            raise HTTPException(status_code=422, detail="body must contain 'observation': [float, ...]")
        if len(obs) != obs_dim:
            raise HTTPException(
                status_code=422, detail=f"expected observation of length {obs_dim}, got {len(obs)}")
        action = policy.predict(obs)
        return {"action": int(action) if policy.discrete else [float(a) for a in action]}

    return app


app = create_app(os.environ.get("DECISIONRL_MODEL", "/models/policy.onnx"))
