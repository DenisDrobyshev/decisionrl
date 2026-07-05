"""Abstract base class shared by every agent."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..utils.logger import Logger
from .env import Env

__all__ = ["BaseAgent"]


class BaseAgent:
    """Common interface for all agents.

    Every agent implements the same four-method surface, regardless of whether
    it is tabular or deep, on-policy or off-policy, discrete or continuous:

    * :meth:`predict` – map an observation to an action.
    * :meth:`learn` – train for a number of environment steps.
    * :meth:`save` / :meth:`load` – round-trip the full agent to disk.
    """

    def __init__(
        self,
        env: Env,
        *,
        seed: Optional[int] = None,
        logger: Optional[Logger] = None,
    ) -> None:
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.logger = logger if logger is not None else Logger()
        self.num_timesteps = 0

    # -- to be implemented by subclasses -----------------------------------
    def predict(self, obs, deterministic: bool = True):
        """Return an action for ``obs`` (single, un-batched observation)."""
        raise NotImplementedError

    def learn(self, total_steps: int, callback: Optional[Any] = None, **kwargs) -> "BaseAgent":
        """Train the agent for ``total_steps`` environment steps."""
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str, env: Optional[Env] = None, **kwargs) -> "BaseAgent":
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}(env={type(self.env).__name__})"
