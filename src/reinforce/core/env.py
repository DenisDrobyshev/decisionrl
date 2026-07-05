"""Environment base class and wrapper, following the Gymnasium step API.

The contract is intentionally identical to Gymnasium so that any
:class:`gymnasium.Env` can be used with this library's agents and vice versa:

* ``reset(*, seed=None, options=None) -> (observation, info)``
* ``step(action) -> (observation, reward, terminated, truncated, info)``

``terminated`` and ``truncated`` are kept separate on purpose. ``terminated``
means the MDP reached a terminal state (bootstrapping should stop); ``truncated``
means an external condition ended the episode early, e.g. a time limit
(bootstrapping should continue from the final observation).
"""

from __future__ import annotations

from typing import Any, Dict, Generic, Optional, Tuple, TypeVar

from .spaces import Space

__all__ = ["Env", "Wrapper"]

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class Env(Generic[ObsType, ActType]):
    """Base class for environments."""

    observation_space: Space
    action_space: Space
    metadata: Dict[str, Any] = {"render_modes": []}

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[ObsType, Dict[str, Any]]:
        """Start a new episode and return ``(observation, info)``."""
        raise NotImplementedError

    def step(
        self, action: ActType
    ) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        """Apply ``action`` and return ``(obs, reward, terminated, truncated, info)``."""
        raise NotImplementedError

    def render(self):  # pragma: no cover - optional
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - optional
        pass

    @property
    def unwrapped(self) -> "Env":
        return self

    def __enter__(self) -> "Env":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class Wrapper(Env[ObsType, ActType]):
    """Wraps an environment to modify its behaviour, forwarding by default."""

    def __init__(self, env: Env) -> None:
        self.env = env
        self._observation_space: Optional[Space] = None
        self._action_space: Optional[Space] = None
        self.metadata = getattr(env, "metadata", {"render_modes": []})

    @property
    def observation_space(self) -> Space:
        return self._observation_space if self._observation_space is not None else self.env.observation_space

    @observation_space.setter
    def observation_space(self, space: Space) -> None:
        self._observation_space = space

    @property
    def action_space(self) -> Space:
        return self._action_space if self._action_space is not None else self.env.action_space

    @action_space.setter
    def action_space(self, space: Space) -> None:
        self._action_space = space

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        return self.env.step(action)

    def render(self):
        return self.env.render()

    def close(self) -> None:
        return self.env.close()

    @property
    def unwrapped(self) -> Env:
        return self.env.unwrapped

    def __getattr__(self, name: str):
        # Forward unknown attribute lookups to the wrapped env.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.env, name)
