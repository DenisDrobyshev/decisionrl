"""Multi-agent RL: environment interface, games, and PPO-based learners."""

from .env import CoordinationGame, MultiAgentEnv, MultiAgentGridWorld, RockPaperScissors
from .ippo import MultiAgentPPO
from .pettingzoo import PettingZooParallelAdapter, make_pettingzoo

__all__ = [
    "MultiAgentEnv",
    "RockPaperScissors",
    "CoordinationGame",
    "MultiAgentGridWorld",
    "MultiAgentPPO",
    "PettingZooParallelAdapter",
    "make_pettingzoo",
]
