"""Optional Gymnasium interoperability.

Gymnasium environments already follow the same ``reset``/``step`` contract as
this library, so agents can consume them directly. :class:`GymAdapter` simply
converts the ``gymnasium.spaces`` into this library's own :class:`Box`/
:class:`Discrete` for a uniform experience, and :func:`make_gym` is a thin
convenience constructor. Gymnasium is an *optional* dependency
(``pip install "decisionrl[gym]"``).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ..core.env import Env
from ..core.spaces import Box, Discrete, Space, is_discrete

__all__ = ["GymAdapter", "make_gym", "make_gym_vec", "make_atari", "make_minigrid",
           "convert_space", "to_gymnasium_space", "to_gymnasium", "register_envs"]


def to_gymnasium_space(space: Space):
    """Convert a decisionrl :class:`Space` into a ``gymnasium.spaces`` object."""
    import gymnasium.spaces as gsp

    if is_discrete(space):
        return gsp.Discrete(int(space.n), start=int(getattr(space, "start", 0)))
    return gsp.Box(low=np.asarray(space.low), high=np.asarray(space.high),
                   shape=space.shape, dtype=space.dtype)


def to_gymnasium(env: Env):
    """Wrap a decisionrl env as a genuine ``gymnasium.Env`` (with gymnasium spaces).

    The inverse of :class:`GymAdapter` — lets Gymnasium users and tooling consume
    decisionrl's environments directly.
    """
    import gymnasium as gym

    class _DecisionrlGymEnv(gym.Env):
        metadata = {"render_modes": ["rgb_array"]}

        def __init__(self) -> None:
            self._env = env
            self.observation_space = to_gymnasium_space(env.observation_space)
            self.action_space = to_gymnasium_space(env.action_space)

        def reset(self, *, seed=None, options=None):
            return self._env.reset(seed=seed, options=options)

        def step(self, action):
            return self._env.step(action)

        def render(self):
            return self._env.render_rgb() if hasattr(self._env, "render_rgb") else None

    return _DecisionrlGymEnv()


def register_envs(prefix: str = "decisionrl", version: str = "v0") -> list:
    """Register all built-in envs with Gymnasium (``gymnasium.make("decisionrl/Inventory-v0")``).

    Returns the list of registered ids. Idempotent; requires ``gymnasium``.
    """
    import gymnasium as gym

    from ..registry import ENVIRONMENTS

    registered = []
    for name, factory in ENVIRONMENTS.items():
        env_id = f"{prefix}/{name}-{version}"
        if env_id in gym.registry:
            continue
        gym.register(id=env_id, entry_point=lambda factory=factory, **kw: to_gymnasium(factory(**kw)))
        registered.append(env_id)
    return registered


def convert_space(space) -> Space:
    """Convert a ``gymnasium.spaces`` object into a decisionrl :class:`Space`."""
    cls_name = type(space).__name__
    if cls_name == "Discrete":
        return Discrete(int(space.n), start=int(getattr(space, "start", 0)))
    if cls_name == "Box":
        return Box(space.low, space.high, shape=space.shape, dtype=space.dtype)
    raise TypeError(
        f"Unsupported gymnasium space {cls_name!r}; only Discrete and Box are supported."
    )


class GymAdapter(Env):
    """Wrap a ``gymnasium.Env`` so it exposes decisionrl spaces."""

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
            "Gymnasium is required for make_gym. Install with: pip install 'decisionrl[gym]'"
        ) from exc

    env = gym.make(env_id, **kwargs)
    return GymAdapter(env) if adapter else env  # type: ignore[return-value]


def make_atari(env_id: str, n_stack: int = 4, screen_size: int = 84, adapter: bool = True, **kwargs):
    """Create an Atari environment with the standard DQN preprocessing.

    Applies grayscale + resize-to-84x84 + frame-skip (via Gymnasium's
    ``AtariPreprocessing``) and stacks ``n_stack`` frames, yielding an
    ``(n_stack, screen_size, screen_size)`` observation ready for the library's
    CNN. Requires ``pip install "gymnasium[atari]" ale-py``.

        from decisionrl.envs import make_atari
        from decisionrl.algorithms import DQN
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

        from decisionrl.envs import make_minigrid
        from decisionrl.algorithms import PPO
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
    combines them with decisionrl's own :class:`~decisionrl.wrappers.SyncVectorEnv`
    / :class:`~decisionrl.wrappers.AsyncVectorEnv`. This deliberately uses
    decisionrl's vectorization (with correct immediate-autoreset and
    ``final_observation`` bootstrapping) rather than ``gymnasium.vector`` so
    behaviour is correct and stable across Gymnasium autoreset-API changes.
    """
    from functools import partial

    from ..wrappers import AsyncVectorEnv, SyncVectorEnv

    fn = partial(make_gym, env_id, **kwargs)
    fns = [fn for _ in range(int(num_envs))]
    return AsyncVectorEnv(fns) if asynchronous else SyncVectorEnv(fns)
