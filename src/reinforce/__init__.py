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
    CQL,
    DDPG,
    DQN,
    IMPALA,
    IQL,
    MBPO,
    PPO,
    QRDQN,
    REINFORCE,
    SAC,
    SARSA,
    TD3,
    TD3BC,
    Dreamer,
    DynaQ,
    ExpectedSARSA,
    QLearning,
    Rainbow,
    RecurrentPPO,
    SACDiscrete,
)
from .core import Box, Dict, Discrete, Env, Space, Transition, Wrapper
from .data import TransitionDataset, collect_dataset
from .distributed import DistributedActorLearner
from .registry import list_algorithms, list_environments, make_agent, make_env, make_vec_env
from .training import evaluate_policy
from .tuning import optuna_search
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
    "Dict",
    "Transition",
    # algorithms
    "QLearning",
    "SARSA",
    "ExpectedSARSA",
    "DynaQ",
    "DQN",
    "C51",
    "QRDQN",
    "Rainbow",
    "REINFORCE",
    "A2C",
    "PPO",
    "IMPALA",
    "RecurrentPPO",
    "DDPG",
    "TD3",
    "SAC",
    "SACDiscrete",
    "TD3BC",
    "IQL",
    "CQL",
    "MBPO",
    "Dreamer",
    # offline data
    "TransitionDataset",
    "collect_dataset",
    "DistributedActorLearner",
    # helpers
    "evaluate_policy",
    "set_seed",
    "make_agent",
    "make_env",
    "make_vec_env",
    "list_algorithms",
    "list_environments",
    "optuna_search",
]
