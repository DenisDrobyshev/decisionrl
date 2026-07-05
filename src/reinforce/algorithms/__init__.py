"""Reinforcement learning algorithms.

Tabular: :class:`QLearning`, :class:`SARSA`, :class:`ExpectedSARSA`.
Value-based deep: :class:`DQN` (Double / Dueling / PER).
Policy gradient: :class:`REINFORCE`, :class:`A2C`, :class:`PPO`.
Continuous control: :class:`DDPG`, :class:`TD3`, :class:`SAC`.
"""

from .a2c import A2C
from .base import OnPolicyAgent
from .ddpg import DDPG
from .dqn import DQN
from .off_policy import OffPolicyContinuousAgent
from .ppo import PPO
from .reinforce import REINFORCE
from .sac import SAC
from .tabular import SARSA, ExpectedSARSA, QLearning
from .td3 import TD3

__all__ = [
    "QLearning",
    "SARSA",
    "ExpectedSARSA",
    "DQN",
    "REINFORCE",
    "A2C",
    "PPO",
    "DDPG",
    "TD3",
    "SAC",
    "OnPolicyAgent",
    "OffPolicyContinuousAgent",
]
