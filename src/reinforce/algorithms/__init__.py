"""Reinforcement learning algorithms.

Tabular: :class:`QLearning`, :class:`SARSA`, :class:`ExpectedSARSA`.
Value-based deep: :class:`DQN` (Double / Dueling / PER).
Policy gradient: :class:`REINFORCE`, :class:`A2C`, :class:`PPO`.
Continuous control: :class:`DDPG`, :class:`TD3`, :class:`SAC`.
Offline: :class:`TD3BC`.
"""

from .a2c import A2C
from .base import OnPolicyAgent
from .c51 import C51
from .cql import CQL
from .ddpg import DDPG
from .decision_transformer import DecisionTransformer
from .diffusion_policy import DiffusionPolicy
from .dqn import DQN
from .dreamer import Dreamer
from .grpo import GRPO
from .impala import IMPALA
from .iql import IQL
from .mbpo import MBPO
from .off_policy import OffPolicyContinuousAgent
from .ppo import PPO
from .qrdqn import QRDQN
from .rainbow import Rainbow
from .recurrent_ppo import RecurrentPPO
from .reinforce import REINFORCE
from .rssm import RSSM, DreamerRSSM
from .sac import SAC
from .sac_discrete import SACDiscrete
from .tabular import SARSA, DynaQ, ExpectedSARSA, QLearning
from .td3 import TD3
from .td3bc import TD3BC

__all__ = [
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
    "MBPO",
    "Dreamer",
    "DreamerRSSM",
    "RSSM",
    "OnPolicyAgent",
    "OffPolicyContinuousAgent",
]
