"""Registries and factories for algorithms and environments.

Enables string-based construction (used by the CLI and configs)::

    from reinforce import make_agent, make_env
    env = make_env("CartPole")
    agent = make_agent("ppo", env, seed=0)

Environment names may also target Gymnasium via a ``"gym:"`` prefix, e.g.
``make_env("gym:LunarLander-v2")``.
"""

from __future__ import annotations

from functools import partial
from typing import Callable, Dict, Union

from .algorithms import (
    A2C,
    C51,
    DDPG,
    DQN,
    IMPALA,
    MBPO,
    PPO,
    QRDQN,
    REINFORCE,
    SAC,
    SARSA,
    TD3,
    Dreamer,
    DynaQ,
    ExpectedSARSA,
    QLearning,
    Rainbow,
    RecurrentPPO,
    SACDiscrete,
)
from .core.agent import BaseAgent
from .core.env import Env
from .envs import (
    Acrobot,
    CartPole,
    GridWorld,
    InventoryManagement,
    MountainCar,
    MountainCarContinuous,
    MultiArmedBandit,
    Pendulum,
    PointMass,
    Thermostat,
)

__all__ = [
    "ALGORITHMS",
    "ENVIRONMENTS",
    "make_agent",
    "make_env",
    "make_vec_env",
    "list_algorithms",
    "list_environments",
]

ALGORITHMS: Dict[str, type] = {
    "qlearning": QLearning,
    "sarsa": SARSA,
    "expected_sarsa": ExpectedSARSA,
    "dynaq": DynaQ,
    "dqn": DQN,
    "c51": C51,
    "qrdqn": QRDQN,
    "rainbow": Rainbow,
    "sac_discrete": SACDiscrete,
    "reinforce": REINFORCE,
    "a2c": A2C,
    "ppo": PPO,
    "impala": IMPALA,
    "recurrent_ppo": RecurrentPPO,
    "ddpg": DDPG,
    "td3": TD3,
    "sac": SAC,
    "mbpo": MBPO,
    "dreamer": Dreamer,
}

ENVIRONMENTS: Dict[str, Callable[..., Env]] = {
    "GridWorld": GridWorld,
    "MultiArmedBandit": MultiArmedBandit,
    "CartPole": CartPole,
    "Pendulum": Pendulum,
    "PointMass": PointMass,
    "MountainCar": MountainCar,
    "MountainCarContinuous": MountainCarContinuous,
    "Acrobot": Acrobot,
    "InventoryManagement": InventoryManagement,
    "Thermostat": Thermostat,
}


def make_agent(name: str, env: Env, **kwargs) -> BaseAgent:
    """Construct an agent by (case-insensitive) name."""
    key = name.lower()
    if key not in ALGORITHMS:
        raise KeyError(f"unknown algorithm {name!r}; available: {sorted(ALGORITHMS)}")
    return ALGORITHMS[key](env, **kwargs)


def make_env(name: str, **kwargs) -> Env:
    """Construct an environment by name, or a Gymnasium env via ``gym:<id>``."""
    if name.startswith("gym:"):
        from .envs import make_gym

        return make_gym(name[len("gym:") :], **kwargs)
    if name not in ENVIRONMENTS:
        raise KeyError(f"unknown environment {name!r}; available: {sorted(ENVIRONMENTS)}")
    return ENVIRONMENTS[name](**kwargs)


def make_vec_env(
    env: Union[str, Callable[[], Env]],
    n_envs: int = 1,
    asynchronous: bool = False,
    **kwargs,
):
    """Create a vectorized environment in one call.

    ``env`` is an environment name (or ``gym:<id>``) or a picklable factory.
    With ``asynchronous=True`` each copy runs in its own process
    (:class:`~reinforce.wrappers.AsyncVectorEnv`), otherwise in-process
    (:class:`~reinforce.wrappers.SyncVectorEnv`).
    """
    if isinstance(env, str):
        fn: Callable[[], Env] = partial(make_env, env, **kwargs)
    elif callable(env):
        fn = env
    else:
        raise TypeError("env must be a name (str) or a callable factory")

    fns = [fn for _ in range(int(n_envs))]
    if asynchronous:
        from .wrappers import AsyncVectorEnv

        return AsyncVectorEnv(fns)
    from .wrappers import SyncVectorEnv

    return SyncVectorEnv(fns)


def list_algorithms() -> list:
    return sorted(ALGORITHMS)


def list_environments() -> list:
    return sorted(ENVIRONMENTS)
