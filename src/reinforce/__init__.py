"""reinforce - a dependency-light, correctness-first reinforcement learning foundation.

Quick start
-----------
>>> from reinforce.algorithms import PPO
>>> from reinforce.envs import CartPole
>>> agent = PPO(CartPole(), seed=0)
>>> agent.learn(total_steps=50_000)          # doctest: +SKIP
>>> from reinforce.training import evaluate_policy
>>> mean, std = evaluate_policy(agent, CartPole())   # doctest: +SKIP

Every agent shares the same surface: ``predict`` / ``learn`` / ``save`` / ``load``.
"""

from . import algorithms, buffers, envs, exploration, networks, training, utils, wrappers
from .algorithms import (
    A2C,
    C51,
    DDPG,
    DQN,
    PPO,
    REINFORCE,
    SAC,
    SARSA,
    TD3,
    TD3BC,
    ExpectedSARSA,
    QLearning,
)
from .core import Box, Discrete, Env, Space, Transition, Wrapper
from .data import TransitionDataset, collect_dataset
from .registry import list_algorithms, list_environments, make_agent, make_env
from .training import evaluate_policy
from .utils import set_seed

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # subpackages
    "algorithms",
    "buffers",
    "envs",
    "exploration",
    "networks",
    "training",
    "utils",
    "wrappers",
    # core
    "Env",
    "Wrapper",
    "Space",
    "Box",
    "Discrete",
    "Transition",
    # algorithms
    "QLearning",
    "SARSA",
    "ExpectedSARSA",
    "DQN",
    "C51",
    "REINFORCE",
    "A2C",
    "PPO",
    "DDPG",
    "TD3",
    "SAC",
    "TD3BC",
    # offline data
    "TransitionDataset",
    "collect_dataset",
    # helpers
    "evaluate_policy",
    "set_seed",
    "make_agent",
    "make_env",
    "list_algorithms",
    "list_environments",
]
