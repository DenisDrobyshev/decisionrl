"""Multi-agent environment interface and a Rock-Paper-Scissors game.

The API mirrors the single-agent one but keyed by agent id::

    reset(seed) -> (obs: {id: obs}, info)
    step(actions: {id: action}) -> (obs, rewards, terminateds, truncateds, info)

each returned mapping keyed by the same agent ids as :attr:`agents`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..core.spaces import Box, Discrete, Space

__all__ = ["MultiAgentEnv", "RockPaperScissors", "CoordinationGame"]


class MultiAgentEnv:
    agents: List[str]
    observation_spaces: Dict[str, Space]
    action_spaces: Dict[str, Space]

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        raise NotImplementedError

    def step(self, actions: Dict[str, Any]):
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - optional
        pass


# rock=0 beats scissors=2, scissors=2 beats paper=1, paper=1 beats rock=0
_BEATS = {(0, 2), (2, 1), (1, 0)}


def _payoff(a: int, b: int) -> float:
    if a == b:
        return 0.0
    return 1.0 if (a, b) in _BEATS else -1.0


class RockPaperScissors(MultiAgentEnv):
    """Two-player zero-sum Rock-Paper-Scissors (single-shot episodes).

    Stateless (the observation is a constant dummy), so the unique Nash
    equilibrium is the uniform mixed strategy - a clean target for self-play.
    """

    def __init__(self) -> None:
        self.agents = ["player_0", "player_1"]
        self.observation_spaces = {a: Box(0.0, 1.0, shape=(1,), dtype=np.float32) for a in self.agents}
        self.action_spaces = {a: Discrete(3) for a in self.agents}
        self._rng = np.random.default_rng()

    def _obs(self) -> Dict[str, np.ndarray]:
        return {a: np.zeros(1, dtype=np.float32) for a in self.agents}

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        return self._obs(), {}

    def step(self, actions: Dict[str, Any]):
        a0, a1 = int(actions["player_0"]), int(actions["player_1"])
        r0 = _payoff(a0, a1)
        rewards = {"player_0": r0, "player_1": -r0}
        terminated = dict.fromkeys(self.agents, True)  # single-shot game
        truncated = dict.fromkeys(self.agents, False)
        info = {"actions": (a0, a1)}
        return self._obs(), rewards, terminated, truncated, info


class CoordinationGame(MultiAgentEnv):
    """Cooperative game: all agents get reward 1 iff they pick the same action.

    A clean, reliably-learnable multi-agent task - the agents must converge on a
    common action (independent learners have to break the symmetry themselves).
    """

    def __init__(self, n_agents: int = 2, n_actions: int = 3) -> None:
        self.agents = [f"agent_{i}" for i in range(n_agents)]
        self.n_actions = int(n_actions)
        self.observation_spaces = {a: Box(0.0, 1.0, shape=(1,), dtype=np.float32) for a in self.agents}
        self.action_spaces = {a: Discrete(self.n_actions) for a in self.agents}

    def _obs(self) -> Dict[str, np.ndarray]:
        return {a: np.zeros(1, dtype=np.float32) for a in self.agents}

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        return self._obs(), {}

    def step(self, actions: Dict[str, Any]):
        chosen = [int(actions[a]) for a in self.agents]
        reward = 1.0 if len(set(chosen)) == 1 else 0.0
        rewards = dict.fromkeys(self.agents, reward)
        terminated = dict.fromkeys(self.agents, True)
        truncated = dict.fromkeys(self.agents, False)
        return self._obs(), rewards, terminated, truncated, {}
