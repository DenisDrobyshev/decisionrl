"""Continuous 2-D maze navigation with lidar — a hard-exploration task.

A point robot with momentum must reach a goal on the far side of a walled arena
containing obstacles, using only local **lidar** range sensors plus its own
kinematics and a goal-direction vector. Reward is dense-ish (progress toward the
goal) with a collision penalty and a terminal bonus, but the walls create a
genuine exploration/credit-assignment problem — a natural fit for SAC and for the
curiosity bonuses in :mod:`reinforce.exploration`.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..core.env import Env
from ..core.spaces import Box

__all__ = ["Navigation2D"]


class Navigation2D(Env):
    metadata = {"render_modes": []}

    # Axis-aligned obstacles (x0, y0, x1, y1) forming two offset walls with gaps.
    OBSTACLES = np.array(
        [
            [0.33, 0.00, 0.40, 0.70],
            [0.60, 0.30, 0.67, 1.00],
        ],
        dtype=np.float64,
    )

    def __init__(
        self,
        n_rays: int = 8,
        max_range: float = 0.5,
        max_speed: float = 0.05,
        accel: float = 0.01,
        radius: float = 0.03,
        goal_radius: float = 0.07,
        max_steps: int = 200,
    ) -> None:
        self.n_rays = int(n_rays)
        self.max_range = float(max_range)
        self.max_speed = float(max_speed)
        self.accel = float(accel)
        self.radius = float(radius)
        self.goal_radius = float(goal_radius)
        self.max_steps = int(max_steps)
        self._ray_angles = np.linspace(0, 2 * np.pi, self.n_rays, endpoint=False)

        # obs: pos(2), vel(2), goal-relative(2), lidar(n_rays)
        obs_dim = 6 + self.n_rays
        self.observation_space = Box(-1.0, 1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        self._rng = np.random.default_rng()
        self._pos = np.zeros(2)
        self._vel = np.zeros(2)
        self._goal = np.array([0.9, 0.9])
        self._steps = 0

    # -- geometry ----------------------------------------------------------
    def _in_obstacle(self, p: np.ndarray) -> bool:
        r = self.radius
        for x0, y0, x1, y1 in self.OBSTACLES:
            if x0 - r <= p[0] <= x1 + r and y0 - r <= p[1] <= y1 + r:
                return True
        return False

    def _in_bounds(self, p: np.ndarray) -> bool:
        r = self.radius
        return bool(r <= p[0] <= 1 - r and r <= p[1] <= 1 - r)

    def _raycast(self, origin: np.ndarray, angle: float) -> float:
        d = np.array([np.cos(angle), np.sin(angle)])
        d = np.where(d == 0, 1e-8, d)
        best = self.max_range

        # Distance to leaving the arena [0,1]^2 (origin is inside).
        t1 = (0.0 - origin) / d
        t2 = (1.0 - origin) / d
        t_exit = float(np.min(np.maximum(t1, t2)))
        if 0 < t_exit < best:
            best = t_exit

        # Nearest obstacle entry ahead of the ray.
        for x0, y0, x1, y1 in self.OBSTACLES:
            lo = np.array([x0, y0])
            hi = np.array([x1, y1])
            tl = (lo - origin) / d
            th = (hi - origin) / d
            tmin = float(np.max(np.minimum(tl, th)))
            tmax = float(np.min(np.maximum(tl, th)))
            if tmax >= max(tmin, 0.0) and 0 <= tmin < best:
                best = tmin
        return best / self.max_range

    def _obs(self) -> np.ndarray:
        lidar = np.array([self._raycast(self._pos, a) for a in self._ray_angles], dtype=np.float32)
        goal_vec = np.clip(self._goal - self._pos, -1.0, 1.0)
        return np.concatenate(
            [self._pos * 2 - 1, self._vel / self.max_speed, goal_vec, lidar]
        ).astype(np.float32)

    # -- api ---------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._pos = self._rng.uniform(0.05, 0.25, size=2)
        self._vel = np.zeros(2)
        self._goal = np.array([0.9, 0.9]) + self._rng.uniform(-0.03, 0.03, size=2)
        self._steps = 0
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(2), -1.0, 1.0)
        prev_dist = float(np.linalg.norm(self._goal - self._pos))

        self._vel = np.clip(self._vel + action * self.accel, -self.max_speed, self.max_speed)
        new_pos = self._pos + self._vel

        collided = False
        if not self._in_bounds(new_pos) or self._in_obstacle(new_pos):
            collided = True
            self._vel = np.zeros(2)  # bump: stop, stay put
        else:
            self._pos = new_pos

        self._steps += 1
        dist = float(np.linalg.norm(self._goal - self._pos))
        reward = 10.0 * (prev_dist - dist) - 0.01
        if collided:
            reward -= 0.1
        terminated = dist < self.goal_radius
        if terminated:
            reward += 10.0
        truncated = self._steps >= self.max_steps and not terminated
        return self._obs(), reward, terminated, truncated, {"distance": dist, "collided": collided}

    def render_rgb(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from ..utils.render import fig_to_rgb

        fig, ax = plt.subplots(figsize=(3.4, 3.4), dpi=64)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, color="#f8fafc", ec="#334155", lw=1.5))
        for x0, y0, x1, y1 in self.OBSTACLES:
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, color="#334155"))
        # lidar rays from the agent
        for a in self._ray_angles:
            d = self._raycast(self._pos, a) * self.max_range
            ax.plot([self._pos[0], self._pos[0] + d * np.cos(a)],
                    [self._pos[1], self._pos[1] + d * np.sin(a)],
                    color="#93c5fd", lw=0.6, alpha=0.7)
        ax.add_patch(plt.Circle(tuple(self._goal), self.goal_radius, color="#16a34a", alpha=0.4))
        ax.plot(*self._goal, marker="*", color="#16a34a", ms=16)
        ax.plot(*self._pos, marker="o", color="#2563eb", ms=10)
        frame = fig_to_rgb(fig)
        plt.close(fig)
        return frame
