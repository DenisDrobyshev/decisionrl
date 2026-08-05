"""A tiny model zoo: save, list and load pretrained (ONNX) policies.

Store exported policies in a directory (default: the repo's ``zoo/``, or set
``REINFORCE_ZOO``) and load them back for inference with only onnxruntime — no
training and no PyTorch needed at load time.

    from decisionrl.zoo import list_pretrained, load_pretrained
    policy = load_pretrained("cartpole-ppo")
    action = policy.predict(obs)

The same policies can be shared through the Hugging Face Hub. :func:`push_to_hub`
uploads an exported policy with its metadata and a generated model card, and
:func:`load_from_hub` downloads one back as an :class:`~decisionrl.serving.OnnxPolicy`.
Both require the optional ``huggingface_hub`` dependency (``pip install decisionrl[hub]``).
"""

from __future__ import annotations

import glob
import json
import os
import tempfile
from typing import List, Optional

from .serving import OnnxPolicy, export_onnx

__all__ = [
    "default_zoo_dir",
    "list_pretrained",
    "load_pretrained",
    "save_to_zoo",
    "push_to_hub",
    "load_from_hub",
]


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
    """Load a pretrained policy by name as an :class:`~decisionrl.serving.OnnxPolicy`."""
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


# -- Hugging Face Hub -------------------------------------------------------------
_ACTION_DESC = {"discrete": "discrete", "continuous": "continuous (box)"}


def model_card(repo_id: str, filename: str, meta: dict, extra: Optional[str] = None) -> str:
    """Build the Markdown model card for a Hub upload (pure; no network access)."""
    action_type = meta.get("action_type", "unknown")
    action_line = _ACTION_DESC.get(action_type, action_type)
    body = f"""---
library_name: decisionrl
tags:
- reinforcement-learning
- decisionrl
- onnx
---

# {repo_id}

A deterministic control policy exported from
[decisionrl](https://github.com/DrobyshevDev/decisionrl) to ONNX. It runs with
`onnxruntime` alone; PyTorch is not required for inference.

## Policy

| Field | Value |
| --- | --- |
| Observation dimension | {meta.get("obs_dim", "?")} |
| Action space | {action_line} |
| File | `{filename}` (+ `{filename}.json` metadata) |

## Usage

```python
from decisionrl.zoo import load_from_hub

policy = load_from_hub("{repo_id}")
action = policy.predict(obs)
```
"""
    if extra:
        body += "\n" + extra.strip() + "\n"
    return body


def push_to_hub(
    agent,
    repo_id: str,
    *,
    token: Optional[str] = None,
    private: bool = False,
    filename: str = "policy.onnx",
    commit_message: str = "Add decisionrl policy",
    extra_card: Optional[str] = None,
) -> str:
    """Export ``agent`` to ONNX and upload it, its metadata and a model card to ``repo_id``.

    Returns the repository URL. Requires ``pip install decisionrl[hub]``.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "push_to_hub requires the 'huggingface_hub' package: pip install decisionrl[hub]"
        ) from exc

    api = HfApi(token=token)
    api.create_repo(repo_id, private=private, exist_ok=True, repo_type="model")
    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = os.path.join(tmp, filename)
        export_onnx(agent, onnx_path)  # writes <filename> and <filename>.json
        with open(onnx_path + ".json", encoding="utf-8") as f:
            meta = json.load(f)
        with open(os.path.join(tmp, "README.md"), "w", encoding="utf-8") as f:
            f.write(model_card(repo_id, filename, meta, extra_card))
        api.upload_folder(repo_id=repo_id, folder_path=tmp, commit_message=commit_message)
    return f"https://huggingface.co/{repo_id}"


def load_from_hub(
    repo_id: str,
    *,
    filename: str = "policy.onnx",
    token: Optional[str] = None,
    revision: Optional[str] = None,
) -> OnnxPolicy:
    """Download an exported policy from the Hub and return an :class:`OnnxPolicy`.

    Requires ``pip install decisionrl[hub]``.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "load_from_hub requires the 'huggingface_hub' package: pip install decisionrl[hub]"
        ) from exc

    kw = {"token": token, "revision": revision}
    # Fetch the metadata sidecar first so it lands in the same snapshot directory
    # that OnnxPolicy will read it from, then the model itself.
    hf_hub_download(repo_id, filename + ".json", **kw)
    onnx_path = hf_hub_download(repo_id, filename, **kw)
    return OnnxPolicy(onnx_path)
