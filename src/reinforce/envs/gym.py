"""Optional Gymnasium interoperability.

Gymnasium environments already follow the same ``reset``/``step`` contract as
this library, so agents can consume them directly. :class:`GymAdapter` simply
converts the ``gymnasium.spaces`` into this library's own :class:`Box`/
:class:`Discrete` for a uniform experience, and :func:`make_gym` is a thin
convenience constructor. Gymnasium is an *optional* dependency
(``pip install "reinforce[gym]"``).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ..core.env import Env
from ..core.spaces import Box, Discrete, Space

__all__ = ["GymAdapter", "make_gym", "convert_space"]


def convert_space(space) -> Space:
    """Convert a ``gymnasium.spaces`` object into a reinforce :class:`Space`."""
    cls_name = type(space).__name__
    if cls_name == "Discrete":
        return Discrete(int(space.n), start=int(getattr(space, "start", 0)))
    if cls_name == "Box":
        return Box(space.low, space.high, shape=space.shape, dtype=space.dtype)
    raise TypeError(
        f"Unsupported gymnasium space {cls_name!r}; only Discrete and Box are supported."
    )


class GymAdapter(Env):
    """Wrap a ``gymnasium.Env`` so it exposes reinforce spaces."""

    def __init__(self, env) -> None:
        self.env = env
        self.observation_space = convert_space(env.observation_space)
        self.action_space = convert_space(env.action_space)
        self.metadata = getattr(env, "metadata", {"render_modes": []})

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        return np.asarray(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return np.asarray(obs), float(reward), bool(terminated), bool(truncated), info

    def render(self):
        return self.env.render()

    def close(self) -> None:
        self.env.close()

    @property
    def unwrapped(self):
        return self.env.unwrapped


def make_gym(env_id: str, adapter: bool = True, **kwargs: Any) -> Env:
    """Create a Gymnasium environment by id.

    Parameters
    ----------
    env_id:
        The Gymnasium id, e.g. ``"CartPole-v1"``.
    adapter:
        If ``True`` (default) wrap it in :class:`GymAdapter` for uniform spaces.
    """
    try:
        import gymnasium as gym
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Gymnasium is required for make_gym. Install with: pip install 'reinforce[gym]'"
        ) from exc

    env = gym.make(env_id, **kwargs)
    return GymAdapter(env) if adapter else env  # type: ignore[return-value]
