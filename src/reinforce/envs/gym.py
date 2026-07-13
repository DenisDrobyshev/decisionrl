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

__all__ = ["GymAdapter", "make_gym", "make_gym_vec", "make_atari", "make_minigrid", "convert_space"]


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


def make_atari(env_id: str, n_stack: int = 4, screen_size: int = 84, adapter: bool = True, **kwargs):
    """Create an Atari environment with the standard DQN preprocessing.

    Applies grayscale + resize-to-84x84 + frame-skip (via Gymnasium's
    ``AtariPreprocessing``) and stacks ``n_stack`` frames, yielding an
    ``(n_stack, screen_size, screen_size)`` observation ready for the library's
    CNN. Requires ``pip install "gymnasium[atari]" ale-py``.

        from reinforce.envs import make_atari
        from reinforce.algorithms import DQN
        agent = DQN(make_atari("ALE/Pong-v5"), seed=0)
    """
    try:
        import gymnasium as gym
        from gymnasium.wrappers import AtariPreprocessing
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Atari support requires: pip install 'gymnasium[atari]' ale-py"
        ) from exc

    env = gym.make(env_id, frameskip=1, **kwargs)  # AtariPreprocessing does the frame-skip
    env = AtariPreprocessing(env, screen_size=screen_size, grayscale_obs=True, scale_obs=False)
    try:  # Gymnasium >= 1.0
        from gymnasium.wrappers import FrameStackObservation as _FrameStack
    except ImportError:  # pragma: no cover - older Gymnasium
        from gymnasium.wrappers import FrameStack as _FrameStack  # type: ignore[no-redef]
    env = _FrameStack(env, n_stack)
    return GymAdapter(env) if adapter else env  # type: ignore[return-value]


def make_minigrid(env_id: str, flatten: bool = True, adapter: bool = True, **kwargs):
    """Create a MiniGrid navigation environment.

    With ``flatten=True`` (default) applies MiniGrid's ``FlatObsWrapper`` to expose
    a flat vector observation usable by any MLP agent; otherwise ``ImgObsWrapper``
    for the image grid (pair with the CNN). Requires ``pip install minigrid``.

        from reinforce.envs import make_minigrid
        from reinforce.algorithms import PPO
        agent = PPO(make_minigrid("MiniGrid-Empty-5x5-v0"), seed=0)
    """
    try:
        import gymnasium as gym
        from minigrid.wrappers import FlatObsWrapper, ImgObsWrapper
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("MiniGrid support requires: pip install minigrid") from exc

    env = gym.make(env_id, **kwargs)
    env = FlatObsWrapper(env) if flatten else ImgObsWrapper(env)
    return GymAdapter(env) if adapter else env  # type: ignore[return-value]


def make_gym_vec(env_id: str, num_envs: int = 1, asynchronous: bool = False, **kwargs):
    """Vectorize a Gymnasium environment for use with this library's agents.

    Builds ``num_envs`` Gymnasium environments wrapped in :class:`GymAdapter` and
    combines them with reinforce's own :class:`~reinforce.wrappers.SyncVectorEnv`
    / :class:`~reinforce.wrappers.AsyncVectorEnv`. This deliberately uses
    reinforce's vectorization (with correct immediate-autoreset and
    ``final_observation`` bootstrapping) rather than ``gymnasium.vector`` so
    behaviour is correct and stable across Gymnasium autoreset-API changes.
    """
    from functools import partial

    from ..wrappers import AsyncVectorEnv, SyncVectorEnv

    fn = partial(make_gym, env_id, **kwargs)
    fns = [fn for _ in range(int(num_envs))]
    return AsyncVectorEnv(fns) if asynchronous else SyncVectorEnv(fns)
