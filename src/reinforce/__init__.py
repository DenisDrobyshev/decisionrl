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

from . import (
    algorithms,
    alphazero,
    buffers,
    dashboard,
    envs,
    evolution,
    exploration,
    networks,
    text,
    training,
    utils,
    wrappers,
)
from .algorithms import (
    A2C,
    C51,
    CQL,
    DDPG,
    DQN,
    GRPO,
    HERDQN,
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
    DecisionTransformer,
    DiffusionPolicy,
    Dreamer,
    DreamerRSSM,
    DynaQ,
    ExpectedSARSA,
    QLearning,
    Rainbow,
    RecurrentPPO,
    SACDiscrete,
)
from .core import Box, Dict, Discrete, Env, Space, Transition, Wrapper
from .data import (
    TrajectoryDataset,
    TransitionDataset,
    collect_dataset,
    collect_trajectories,
)
from .distributed import DistributedActorLearner
from .evaluation import (
    aggregate_metrics,
    bootstrap_ci,
    iqm,
    performance_profile,
    probability_of_improvement,
    run_seeds,
)
from .evolution import NeuroevolutionAgent
from .imitation import BC, GAIL, DAgger, GAILDiscriminator, collect_expert_dataset
from .registry import list_algorithms, list_environments, make_agent, make_env, make_vec_env
from .rlhf import (
    DPO,
    PreferenceDataset,
    RewardModel,
    RewardModelWrapper,
    collect_segments,
    synthetic_preferences,
    train_reward_model,
)
from .training import evaluate_policy
from .tuning import optuna_search
from .utils import set_seed
from .zoo import list_pretrained, load_pretrained, save_to_zoo

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # subpackages
    "algorithms",
    "alphazero",
    "buffers",
    "dashboard",
    "envs",
    "evolution",
    "exploration",
    "networks",
    "text",
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
    "GRPO",
    "IMPALA",
    "RecurrentPPO",
    "DDPG",
    "TD3",
    "SAC",
    "SACDiscrete",
    "TD3BC",
    "IQL",
    "CQL",
    "DecisionTransformer",
    "DiffusionPolicy",
    "HERDQN",
    "MBPO",
    "Dreamer",
    "DreamerRSSM",
    "NeuroevolutionAgent",
    # offline data
    "TransitionDataset",
    "collect_dataset",
    "TrajectoryDataset",
    "collect_trajectories",
    "DistributedActorLearner",
    # RLHF
    "RewardModel",
    "PreferenceDataset",
    "collect_segments",
    "synthetic_preferences",
    "train_reward_model",
    "RewardModelWrapper",
    "DPO",
    # imitation learning
    "BC",
    "DAgger",
    "GAIL",
    "GAILDiscriminator",
    "collect_expert_dataset",
    # reliable evaluation statistics
    "iqm",
    "bootstrap_ci",
    "aggregate_metrics",
    "performance_profile",
    "probability_of_improvement",
    "run_seeds",
    # helpers
    "evaluate_policy",
    "set_seed",
    "make_agent",
    "make_env",
    "make_vec_env",
    "list_algorithms",
    "list_environments",
    "optuna_search",
    # model zoo
    "list_pretrained",
    "load_pretrained",
    "save_to_zoo",
]
