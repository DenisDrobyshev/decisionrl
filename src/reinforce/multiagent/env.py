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

__all__ = ["MultiAgentEnv", "RockPaperScissors", "CoordinationGame", "MultiAgentGridWorld"]

_MA_DELTAS = [(-1, 0), (0, 1), (1, 0), (0, -1), (0, 0)]  # up, right, down, left, stay


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


class MultiAgentGridWorld(MultiAgentEnv):
    """Cooperative multi-step navigation: each agent must reach its own target.

    Multi-step (unlike the single-shot games above): a dense per-agent reward
    (negative normalized distance) plus a bonus for reaching the target, so the
    agents learn temporally-extended navigation. Each agent's observation is its
    own ``[position, target]`` (normalized), so a single shared policy generalizes
    across agents.
    """

    def __init__(self, rows: int = 5, cols: int = 5, n_agents: int = 2, horizon: int = 25) -> None:
        self.rows, self.cols, self.horizon = int(rows), int(cols), int(horizon)
        self.agents = [f"agent_{i}" for i in range(n_agents)]
        corners = [(0, 0), (rows - 1, cols - 1), (0, cols - 1), (rows - 1, 0)]
        self.targets = {a: corners[i % len(corners)] for i, a in enumerate(self.agents)}
        self.observation_spaces = {a: Box(0.0, 1.0, shape=(4,), dtype=np.float32) for a in self.agents}
        self.action_spaces = {a: Discrete(5) for a in self.agents}
        self._rng = np.random.default_rng()
        self._pos: Dict[str, tuple] = {}
        self._steps = 0

    def _obs_for(self, a: str) -> np.ndarray:
        r, c = self._pos[a]
        tr, tc = self.targets[a]
        return np.array([r / (self.rows - 1), c / (self.cols - 1),
                         tr / (self.rows - 1), tc / (self.cols - 1)], dtype=np.float32)

    def _obs(self) -> Dict[str, np.ndarray]:
        return {a: self._obs_for(a) for a in self.agents}

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._pos = {a: (int(self._rng.integers(self.rows)), int(self._rng.integers(self.cols)))
                     for a in self.agents}
        self._steps = 0
        return self._obs(), {}

    def step(self, actions: Dict[str, Any]):
        rewards = {}
        max_dist = (self.rows - 1) + (self.cols - 1)
        for a in self.agents:
            dr, dc = _MA_DELTAS[int(actions[a])]
            r, c = self._pos[a]
            self._pos[a] = (int(np.clip(r + dr, 0, self.rows - 1)), int(np.clip(c + dc, 0, self.cols - 1)))
            tr, tc = self.targets[a]
            dist = abs(self._pos[a][0] - tr) + abs(self._pos[a][1] - tc)
            reached = self._pos[a] == self.targets[a]
            rewards[a] = (1.0 if reached else 0.0) - dist / max_dist
        self._steps += 1
        all_reached = all(self._pos[a] == self.targets[a] for a in self.agents)
        terminated = dict.fromkeys(self.agents, all_reached)
        truncated = dict.fromkeys(self.agents, self._steps >= self.horizon and not all_reached)
        return self._obs(), rewards, terminated, truncated, {}
