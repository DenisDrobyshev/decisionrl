"""A small, configurable grid-world MDP (tabular, dependency-free)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.env import Env
from ..core.spaces import Box, Discrete

__all__ = ["GridWorld"]

# actions: 0=up, 1=right, 2=down, 3=left
_DELTAS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


class GridWorld(Env):
    """A gridworld where the agent must reach a goal cell.

    Observations are the integer index of the current cell (``Discrete``), which
    makes the environment ideal for tabular methods. Set ``one_hot=True`` to get
    one-hot ``Box`` observations suitable for function approximation instead.

    Rewards: ``goal_reward`` upon reaching the goal (episode terminates), and
    ``-step_penalty`` on every other step, encouraging short paths. Walls block
    movement; ``slip_prob`` adds stochasticity by occasionally moving the agent
    perpendicular to the intended direction.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        rows: int = 4,
        cols: int = 4,
        start: Tuple[int, int] = (0, 0),
        goal: Optional[Tuple[int, int]] = None,
        walls: Optional[List[Tuple[int, int]]] = None,
        step_penalty: float = 0.01,
        goal_reward: float = 1.0,
        slip_prob: float = 0.0,
        max_steps: int = 100,
        one_hot: bool = False,
    ) -> None:
        self.rows = int(rows)
        self.cols = int(cols)
        self.start = start
        self.goal = goal if goal is not None else (rows - 1, cols - 1)
        self.walls = set(walls or [])
        self.step_penalty = float(step_penalty)
        self.goal_reward = float(goal_reward)
        self.slip_prob = float(slip_prob)
        self.max_steps = int(max_steps)
        self.one_hot = bool(one_hot)

        self.n_states = self.rows * self.cols
        self.action_space = Discrete(4)
        if one_hot:
            self.observation_space = Box(0.0, 1.0, shape=(self.n_states,), dtype=np.float32)
        else:
            self.observation_space = Discrete(self.n_states)

        self._rng = np.random.default_rng()
        self._pos = start
        self._steps = 0

    # -- helpers -----------------------------------------------------------
    def _to_index(self, pos: Tuple[int, int]) -> int:
        return pos[0] * self.cols + pos[1]

    def _obs(self) -> Any:
        idx = self._to_index(self._pos)
        if self.one_hot:
            v = np.zeros(self.n_states, dtype=np.float32)
            v[idx] = 1.0
            return v
        return idx

    def _valid(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols and pos not in self.walls

    # -- gym API -----------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._pos = self.start
        self._steps = 0
        return self._obs(), {}

    def step(self, action: int):
        action = int(action)
        assert self.action_space.contains(action), f"invalid action {action}"

        if self.slip_prob > 0 and self._rng.random() < self.slip_prob:
            # slip 90 degrees left or right of the intended move
            action = (action + self._rng.choice([-1, 1])) % 4

        dr, dc = _DELTAS[action]
        candidate = (self._pos[0] + dr, self._pos[1] + dc)
        if self._valid(candidate):
            self._pos = candidate

        self._steps += 1
        terminated = self._pos == self.goal
        truncated = self._steps >= self.max_steps and not terminated
        reward = self.goal_reward if terminated else -self.step_penalty
        return self._obs(), float(reward), terminated, truncated, {}

    def render(self) -> str:
        rows = []
        for r in range(self.rows):
            line = []
            for c in range(self.cols):
                if (r, c) == self._pos:
                    line.append("A")
                elif (r, c) == self.goal:
                    line.append("G")
                elif (r, c) in self.walls:
                    line.append("#")
                else:
                    line.append(".")
            rows.append(" ".join(line))
        out = "\n".join(rows)
        print(out)
        return out

    def render_rgb(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from ..utils.render import fig_to_rgb

        fig, ax = plt.subplots(figsize=(3, 3), dpi=72)
        ax.set_xlim(-0.5, self.cols - 0.5)
        ax.set_ylim(-0.5, self.rows - 0.5)
        ax.set_aspect("equal")
        ax.set_xticks(range(self.cols))
        ax.set_yticks(range(self.rows))
        ax.grid(True, color="#cbd5e1")
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        def xy(pos):
            return pos[1], self.rows - 1 - pos[0]

        for wall in self.walls:
            wx, wy = xy(wall)
            ax.add_patch(plt.Rectangle((wx - 0.5, wy - 0.5), 1, 1, color="#475569"))
        gx, gy = xy(self.goal)
        ax.add_patch(plt.Rectangle((gx - 0.5, gy - 0.5), 1, 1, color="#16a34a", alpha=0.5))
        ax_x, ax_y = xy(self._pos)
        ax.plot([ax_x], [ax_y], marker="o", color="#2563eb", ms=22)
        frame = fig_to_rgb(fig)
        plt.close(fig)
        return frame
