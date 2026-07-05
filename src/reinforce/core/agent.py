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

    # -- resume: model + optimizer (via save/load) plus training state --------
    def save_checkpoint(self, path: str) -> None:
        """Save a resumable checkpoint: the agent plus ``num_timesteps`` and RNG
        state (written to ``path`` and ``path + ".state"``). Restore with
        :meth:`load_checkpoint`. Note: replay buffers are not persisted.
        """
        import torch

        self.save(path)
        torch.save(
            {
                "num_timesteps": int(self.num_timesteps),
                "agent_rng": self.rng.bit_generator.state,
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
            },
            path + ".state",
        )

    @classmethod
    def load_checkpoint(cls, path: str, env: Optional[Env] = None, **kwargs) -> "BaseAgent":
        """Load an agent saved with :meth:`save_checkpoint` and restore its
        training step count and RNG state, ready to continue training.
        """
        import os

        import torch

        agent = cls.load(path, env=env, **kwargs)
        state_path = path + ".state"
        if os.path.exists(state_path):
            state = torch.load(state_path, weights_only=False)
            agent.num_timesteps = int(state["num_timesteps"])
            agent.rng.bit_generator.state = state["agent_rng"]
            np.random.set_state(state["numpy_rng"])
            torch.set_rng_state(state["torch_rng"])
        return agent

    def __repr__(self) -> str:
        return f"{type(self).__name__}(env={type(self.env).__name__})"
