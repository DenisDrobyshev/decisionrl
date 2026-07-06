"""Multi-agent RL: environment interface, games, and PPO-based learners."""

from .env import CoordinationGame, MultiAgentEnv, RockPaperScissors
from .ippo import MultiAgentPPO

__all__ = ["MultiAgentEnv", "RockPaperScissors", "CoordinationGame", "MultiAgentPPO"]
