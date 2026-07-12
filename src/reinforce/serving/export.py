"""Export a trained agent's deterministic policy to ONNX / TorchScript.

A small ``nn.Module`` is built that maps a batch of observations directly to
deterministic actions (argmax for discrete agents; the squashed/clamped mean for
continuous ones), so the exported graph is a pure feed-forward policy with no
sampling or Python control flow — ideal for portable, low-latency serving.
"""

from __future__ import annotations

import json
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from ..networks.policies import (
    CategoricalActor,
    DeterministicActor,
    GaussianActor,
    SquashedGaussianActor,
)

__all__ = ["build_policy_module", "export_onnx", "export_torchscript", "export_json", "OnnxPolicy"]


class _CategoricalPolicy(nn.Module):
    def __init__(self, actor: CategoricalActor) -> None:
        super().__init__()
        self.net = actor.net

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.net(obs), dim=-1)


class _QPolicy(nn.Module):
    def __init__(self, q_net: nn.Module) -> None:
        super().__init__()
        self.q_net = q_net

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.q_net(obs), dim=-1)


class _GaussianPolicy(nn.Module):
    def __init__(self, actor: GaussianActor, low: np.ndarray, high: np.ndarray) -> None:
        super().__init__()
        self.mean_net = actor.mean_net
        self.register_buffer("low", torch.as_tensor(low, dtype=torch.float32))
        self.register_buffer("high", torch.as_tensor(high, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.mean_net(obs), self.low, self.high)


class _SquashedPolicy(nn.Module):
    def __init__(self, actor: SquashedGaussianActor) -> None:
        super().__init__()
        self.actor = actor

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self.actor._mean_logstd(obs)
        return torch.tanh(mean) * self.actor.action_scale + self.actor.action_bias


class _DeterministicPolicy(nn.Module):
    def __init__(self, actor: DeterministicActor) -> None:
        super().__init__()
        self.actor = actor

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)


def build_policy_module(agent) -> Tuple[nn.Module, dict]:
    """Return ``(module, metadata)`` for a supported agent's deterministic policy.

    Supported: PPO/A2C/GRPO (discrete + continuous), SAC/DDPG/TD3 (continuous)
    and DQN (discrete). Distributional value agents (C51/QR-DQN/Rainbow) are not
    supported for export.
    """
    obs_dim = int(getattr(agent, "obs_dim", int(np.prod(agent.observation_space.shape))))
    actor = getattr(agent, "actor", None)

    if isinstance(actor, CategoricalActor):
        n_actions = int(getattr(agent, "n_actions", actor.net[-1].out_features))
        module, meta = _CategoricalPolicy(actor), {"action_type": "discrete", "n_actions": n_actions}
    elif isinstance(actor, SquashedGaussianActor):
        module, meta = _SquashedPolicy(actor), _continuous_meta(agent)
    elif isinstance(actor, DeterministicActor):
        module, meta = _DeterministicPolicy(actor), _continuous_meta(agent)
    elif isinstance(actor, GaussianActor):
        low = np.asarray(agent.action_space.low, dtype=np.float32)
        high = np.asarray(agent.action_space.high, dtype=np.float32)
        module, meta = _GaussianPolicy(actor, low, high), _continuous_meta(agent)
    elif hasattr(agent, "q_net") and not _is_distributional(agent):
        module = _QPolicy(agent.q_net)
        meta = {"action_type": "discrete", "n_actions": int(agent.n_actions)}
    else:
        raise TypeError(
            f"export not supported for {type(agent).__name__}; supported agents: "
            "PPO, A2C, GRPO, SAC, DDPG, TD3, DQN"
        )
    meta["obs_dim"] = obs_dim
    return module.eval(), meta


def _continuous_meta(agent) -> dict:
    return {
        "action_type": "continuous",
        "action_dim": int(agent.action_space.shape[0]),
        "action_low": np.asarray(agent.action_space.low, dtype=np.float32).tolist(),
        "action_high": np.asarray(agent.action_space.high, dtype=np.float32).tolist(),
    }


def _is_distributional(agent) -> bool:
    return any(hasattr(agent, a) for a in ("n_atoms", "num_atoms", "n_quantiles", "num_quantiles"))


def _write_meta(path: str, meta: dict) -> None:
    with open(path + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def export_onnx(agent, path: str, opset: int = 17) -> str:
    """Export the deterministic policy to ``path`` (ONNX) + ``path + ".json"``."""
    module, meta = build_policy_module(agent)
    device = next(module.parameters()).device
    dummy = torch.zeros(1, meta["obs_dim"], dtype=torch.float32, device=device)
    kwargs = dict(
        input_names=["observation"], output_names=["action"],
        dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
        opset_version=opset,
    )
    try:
        # Use the legacy TorchScript-based exporter (no onnxscript dependency).
        torch.onnx.export(module, dummy, path, dynamo=False, **kwargs)
    except TypeError:  # older torch without the `dynamo` kwarg
        torch.onnx.export(module, dummy, path, **kwargs)
    _write_meta(path, meta)
    return path


def export_json(agent, path: str) -> str:
    """Export the policy MLP weights as plain JSON (for in-browser inference).

    Writes ``{obs_dim, action_type, activation, layers:[{w, b}], ...}`` — enough to
    run the deterministic policy with a few matmuls in any language (e.g. a
    self-contained JavaScript demo), with no PyTorch/ONNX runtime.
    """
    _, meta = build_policy_module(agent)
    net = getattr(agent, "actor", None)
    if net is not None and hasattr(net, "net"):
        sequential = net.net            # CategoricalActor
    elif net is not None and hasattr(net, "mean_net"):
        sequential = net.mean_net       # GaussianActor
    elif hasattr(agent, "q_net"):
        sequential = agent.q_net.net if hasattr(agent.q_net, "net") else agent.q_net
    else:
        raise TypeError(f"export_json not supported for {type(agent).__name__}")

    layers = []
    activation = "tanh"
    for m in sequential:
        if isinstance(m, nn.Linear):
            layers.append({"w": m.weight.detach().cpu().numpy().tolist(),
                           "b": m.bias.detach().cpu().numpy().tolist()})
        elif isinstance(m, nn.ReLU):
            activation = "relu"
    meta = dict(meta)
    meta["activation"] = activation
    meta["layers"] = layers
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return path


def export_torchscript(agent, path: str) -> str:
    """Export the deterministic policy to ``path`` (TorchScript) + metadata."""
    module, meta = build_policy_module(agent)
    device = next(module.parameters()).device
    dummy = torch.zeros(1, meta["obs_dim"], dtype=torch.float32, device=device)
    scripted = torch.jit.trace(module, dummy)
    scripted.save(path)
    _write_meta(path, meta)
    return path


class OnnxPolicy:
    """Load an exported ONNX policy and run inference with onnxruntime + NumPy."""

    def __init__(self, model_path: str) -> None:
        import onnxruntime as ort

        with open(model_path + ".json", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    @property
    def discrete(self) -> bool:
        return self.meta["action_type"] == "discrete"

    def predict(self, obs):
        """Return the deterministic action for a single observation."""
        x = np.asarray(obs, dtype=np.float32).reshape(1, self.meta["obs_dim"])
        out = self.session.run(None, {self.input_name: x})[0]
        if self.discrete:
            return int(np.asarray(out).reshape(-1)[0])
        return np.asarray(out, dtype=np.float32).reshape(-1)
