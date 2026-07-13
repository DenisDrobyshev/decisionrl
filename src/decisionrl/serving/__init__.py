"""Deployment: export trained policies and serve them over HTTP.

Turns a trained agent into a portable, framework-independent artifact and a tiny
inference service:

* :func:`export_onnx` / :func:`export_torchscript` — freeze the deterministic
  policy to ONNX (or TorchScript) plus a JSON metadata sidecar.
* :class:`OnnxPolicy` — load an exported ONNX policy and run inference with only
  ``onnxruntime`` + NumPy (no PyTorch needed at serving time).
* :func:`create_app` — a FastAPI app exposing ``/predict``, ``/health`` and
  ``/info`` for the exported policy (see also ``deploy/Dockerfile``).
"""

from .export import OnnxPolicy, build_policy_module, export_json, export_onnx, export_torchscript

__all__ = ["export_onnx", "export_torchscript", "export_json", "OnnxPolicy",
           "build_policy_module", "create_app"]


def create_app(model_path: str):
    """Lazily build the FastAPI app (keeps FastAPI an optional dependency)."""
    from .server import create_app as _create_app

    return _create_app(model_path)
