"""A tiny model zoo: save, list and load pretrained (ONNX) policies.

Store exported policies in a directory (default: the repo's ``zoo/``, or set
``REINFORCE_ZOO``) and load them back for inference with only onnxruntime — no
training and no PyTorch needed at load time.

    from reinforce.zoo import list_pretrained, load_pretrained
    policy = load_pretrained("cartpole-ppo")
    action = policy.predict(obs)
"""

from __future__ import annotations

import glob
import os
from typing import List, Optional

from .serving import OnnxPolicy, export_onnx

__all__ = ["default_zoo_dir", "list_pretrained", "load_pretrained", "save_to_zoo"]


def default_zoo_dir() -> str:
    env = os.environ.get("REINFORCE_ZOO")
    if env:
        return env
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "zoo"))


def list_pretrained(zoo_dir: Optional[str] = None) -> List[str]:
    """Names of the pretrained policies available in ``zoo_dir``."""
    zoo_dir = zoo_dir or default_zoo_dir()
    if not os.path.isdir(zoo_dir):
        return []
    return sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(zoo_dir, "*.onnx")))


def load_pretrained(name: str, zoo_dir: Optional[str] = None) -> OnnxPolicy:
    """Load a pretrained policy by name as an :class:`~reinforce.serving.OnnxPolicy`."""
    zoo_dir = zoo_dir or default_zoo_dir()
    path = os.path.join(zoo_dir, f"{name}.onnx")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no pretrained model {name!r} in {zoo_dir}; available: {list_pretrained(zoo_dir)}"
        )
    return OnnxPolicy(path)


def save_to_zoo(agent, name: str, zoo_dir: Optional[str] = None) -> str:
    """Export ``agent`` to the zoo as ``<name>.onnx`` (+ metadata)."""
    zoo_dir = zoo_dir or default_zoo_dir()
    os.makedirs(zoo_dir, exist_ok=True)
    return export_onnx(agent, os.path.join(zoo_dir, f"{name}.onnx"))
