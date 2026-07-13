"""A minimal FastAPI service that serves an exported ONNX policy.

    from decisionrl.serving import create_app
    app = create_app("policy.onnx")     # uvicorn decisionrl.serving.server:app

Endpoints:
    GET  /health   -> liveness probe
    GET  /info     -> policy metadata (obs dim, action type, bounds)
    POST /predict  -> {"observation": [...]} -> {"action": ...}
"""

from __future__ import annotations

import os

from .export import OnnxPolicy

__all__ = ["create_app"]


def create_app(model_path: str):
    from fastapi import Body, FastAPI, HTTPException

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
                status_code=422,
                detail=f"expected observation of length {obs_dim}, got {len(obs)}",
            )
        action = policy.predict(obs)
        if policy.discrete:
            return {"action": int(action)}
        return {"action": [float(a) for a in action]}

    return app


# Module-level app for `uvicorn decisionrl.serving.server:app`, driven by env var.
if os.environ.get("DECISIONRL_MODEL"):  # pragma: no cover - runtime entrypoint
    app = create_app(os.environ["DECISIONRL_MODEL"])
