"""Adapter from PettingZoo (parallel API) to this library's ``MultiAgentEnv``.

PettingZoo's Parallel API already matches ``MultiAgentEnv`` almost exactly
(dicts keyed by agent id), so the adapter just converts spaces and normalizes the
reset/step return shapes. Requires ``pip install pettingzoo``.

    from reinforce.multiagent import make_pettingzoo
    from reinforce.multiagent import MultiAgentPPO
    env = make_pettingzoo("pettingzoo.mpe.simple_spread_v3")
    agent = MultiAgentPPO(env, shared_policy=False, seed=0)
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Optional

import numpy as np

from ..envs.gym import convert_space
from .env import MultiAgentEnv

__all__ = ["PettingZooParallelAdapter", "make_pettingzoo"]


class PettingZooParallelAdapter(MultiAgentEnv):
    """Wrap a PettingZoo ``ParallelEnv`` as a :class:`MultiAgentEnv`."""

    def __init__(self, parallel_env) -> None:
        self.env = parallel_env
        self.agents = list(parallel_env.possible_agents)
        self.observation_spaces = {a: convert_space(parallel_env.observation_space(a)) for a in self.agents}
        self.action_spaces = {a: convert_space(parallel_env.action_space(a)) for a in self.agents}

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        return {a: np.asarray(o) for a, o in obs.items()}, info

    def step(self, actions: Dict[str, Any]):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        obs = {a: np.asarray(o) for a, o in obs.items()}
        return obs, rewards, terminations, truncations, infos

    def close(self) -> None:  # pragma: no cover - passthrough
        self.env.close()


def make_pettingzoo(module_path: str, **kwargs) -> PettingZooParallelAdapter:
    """Build a PettingZoo parallel env from ``"package.module"`` and adapt it.

    ``module_path`` is the importable env module (e.g.
    ``"pettingzoo.mpe.simple_spread_v3"``); its ``parallel_env(**kwargs)`` is used.
    """
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("PettingZoo support requires: pip install pettingzoo") from exc
    if not hasattr(module, "parallel_env"):
        raise TypeError(f"{module_path} has no parallel_env(); use a PettingZoo parallel environment")
    return PettingZooParallelAdapter(module.parallel_env(**kwargs))
