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
from .ddpg import DDPG
from .dqn import DQN
from .off_policy import OffPolicyContinuousAgent
from .ppo import PPO
from .reinforce import REINFORCE
from .sac import SAC
from .tabular import SARSA, ExpectedSARSA, QLearning
from .td3 import TD3
from .td3bc import TD3BC

__all__ = [
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
    "OnPolicyAgent",
    "OffPolicyContinuousAgent",
]
