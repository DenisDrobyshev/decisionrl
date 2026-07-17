"""Run an experiment from a config (dict / YAML / JSON) instead of a script.

A config names an environment, an algorithm and a training budget; :func:`build`
constructs the agent + env via the registry, and :func:`run` trains and evaluates
it. This makes experiments declarative and diff-able, and pairs with
:mod:`decisionrl.tracking` to record a reproducible manifest.

Schema (every field optional except ``env`` and ``algo``)::

    env: CartPole                 # name, "gym:<id>", or {name: ..., <kwargs>}
    algo: ppo                     # name, or {name: ..., <hyperparams>}
    seed: 0
    total_steps: 50000
    eval_episodes: 20

YAML needs PyYAML (``pip install decisionrl[config]``); JSON works out of the box.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple, Union

from .core.agent import BaseAgent
from .core.env import Env
from .registry import make_agent, make_env

__all__ = ["load_config", "build", "run"]

ConfigLike = Union[str, Path, Dict[str, Any]]


def load_config(source: ConfigLike) -> Dict[str, Any]:
    """Load a config from a dict, or a ``.yaml``/``.yml``/``.json`` file."""
    if isinstance(source, dict):
        return dict(source)
    path = Path(source)
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "PyYAML is required for YAML configs: pip install 'decisionrl[config]' "
                "(or use a .json config)."
            ) from exc
        return dict(yaml.safe_load(text))
    if path.suffix == ".json":
        return dict(json.loads(text))
    raise ValueError(f"unsupported config format {path.suffix!r}; use .yaml, .yml or .json")


def _split(spec: Union[str, Dict[str, Any]], key: str = "name") -> Tuple[str, Dict[str, Any]]:
    """Normalize ``"ppo"`` or ``{name: ppo, lr: ...}`` to ``("ppo", {kwargs})``."""
    if isinstance(spec, str):
        return spec, {}
    spec = dict(spec)
    name = spec.pop(key)
    return name, spec


def build(config: ConfigLike) -> Tuple[BaseAgent, Env]:
    """Construct ``(agent, env)`` from a config, without training."""
    cfg = load_config(config)
    if "env" not in cfg or "algo" not in cfg:
        raise KeyError("config must define both 'env' and 'algo'")
    env_name, env_kwargs = _split(cfg["env"])
    algo_name, algo_kwargs = _split(cfg["algo"])
    seed = cfg.get("seed")
    env = make_env(env_name, **env_kwargs)
    agent = make_agent(algo_name, env, seed=seed, **algo_kwargs)
    return agent, env


def run(config: ConfigLike, total_steps: int = None) -> Dict[str, Any]:
    """Build, train and evaluate from a config; return a results dict.

    ``total_steps`` overrides the config's value when given. The returned dict
    holds the resolved config, the trained ``agent`` and the eval ``mean``/``std``.
    """
    from .training import evaluate_policy

    cfg = load_config(config)
    agent, env = build(cfg)
    steps = total_steps if total_steps is not None else int(cfg.get("total_steps", 50_000))
    agent.learn(steps)

    result: Dict[str, Any] = {"config": cfg, "agent": agent, "total_steps": steps}
    episodes = int(cfg.get("eval_episodes", 0))
    if episodes:
        env_name, env_kwargs = _split(cfg["env"])
        mean, std = evaluate_policy(agent, make_env(env_name, **env_kwargs),
                                    n_episodes=episodes, seed=cfg.get("seed"))
        result["mean"], result["std"] = mean, std

    # Optional reproducibility manifest: `manifest: path.json` in the config.
    manifest_path = cfg.get("manifest")
    if manifest_path:
        from .tracking import run_manifest, save_manifest
        metrics = {k: result[k] for k in ("mean", "std") if k in result}
        metrics["total_steps"] = steps
        save_manifest(run_manifest(cfg, metrics=metrics, seed=cfg.get("seed")), manifest_path)
        result["manifest"] = manifest_path
    return result
