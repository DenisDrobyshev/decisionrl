"""Two-link planar reaching arm — a continuous robotic-manipulation task.

A 2-DoF arm actuated by joint torques must bring its fingertip to a randomly
placed target and hold it there. The observation encodes joint angles (as
cos/sin to avoid the angle wrap discontinuity), joint velocities and the
fingertip-to-target vector; the reward is a dense negative distance with a small
control-effort penalty. Harder than the toy tasks: an 8-D+ observation, 2-D
continuous control and non-linear kinematics. Solve with SAC or TD3.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

from ..core.env import Env
from ..core.spaces import Box

__all__ = ["ReacherArm"]


class ReacherArm(Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        link1: float = 0.1,
        link2: float = 0.1,
        max_torque: float = 1.0,
        max_speed: float = 10.0,
        dt: float = 0.05,
        damping: float = 0.1,
        max_steps: int = 50,
    ) -> None:
        self.l1 = float(link1)
        self.l2 = float(link2)
        self.max_torque = float(max_torque)
        self.max_speed = float(max_speed)
        self.dt = float(dt)
        self.damping = float(damping)
        self.max_steps = int(max_steps)
        self.reach = self.l1 + self.l2

        # obs: cos/sin of both joints, both joint velocities, target xy, tip-to-target xy
        high = np.array([1, 1, 1, 1, 1, 1, self.reach, self.reach, 2 * self.reach, 2 * self.reach],
                        dtype=np.float32)
        self.observation_space = Box(-high, high, dtype=np.float32)
        self.action_space = Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        self._rng = np.random.default_rng()
        self._theta = np.zeros(2, dtype=np.float32)
        self._theta_dot = np.zeros(2, dtype=np.float32)
        self._target = np.zeros(2, dtype=np.float32)
        self._steps = 0

    def _fingertip(self) -> np.ndarray:
        t1, t2 = self._theta
        elbow = self.l1 * np.array([math.cos(t1), math.sin(t1)])
        tip = elbow + self.l2 * np.array([math.cos(t1 + t2), math.sin(t1 + t2)])
        return tip.astype(np.float32)

    def _obs(self) -> np.ndarray:
        t1, t2 = self._theta
        tip = self._fingertip()
        diff = self._target - tip
        return np.array(
            [math.cos(t1), math.sin(t1), math.cos(t2), math.sin(t2),
             self._theta_dot[0] / self.max_speed, self._theta_dot[1] / self.max_speed,
             self._target[0], self._target[1], diff[0], diff[1]],
            dtype=np.float32,
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._theta = self._rng.uniform(-math.pi, math.pi, size=2).astype(np.float32)
        self._theta_dot = np.zeros(2, dtype=np.float32)
        # Random reachable target (inside the annulus the arm can reach).
        radius = self._rng.uniform(0.05, self.reach * 0.95)
        angle = self._rng.uniform(-math.pi, math.pi)
        self._target = np.array([radius * math.cos(angle), radius * math.sin(angle)], dtype=np.float32)
        self._steps = 0
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(2), -1.0, 1.0)
        torque = action * self.max_torque
        theta_ddot = torque - self.damping * self._theta_dot
        self._theta_dot = np.clip(self._theta_dot + theta_ddot * self.dt, -self.max_speed, self.max_speed)
        self._theta = self._theta + self._theta_dot * self.dt
        self._theta = ((self._theta + math.pi) % (2 * math.pi) - math.pi).astype(np.float32)
        self._steps += 1

        distance = float(np.linalg.norm(self._target - self._fingertip()))
        reward = -distance - 0.01 * float(np.sum(action**2))
        terminated = False
        truncated = self._steps >= self.max_steps
        return self._obs(), reward, terminated, truncated, {"distance": distance}

    def render_rgb(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from ..utils.render import fig_to_rgb

        t1, t2 = self._theta
        elbow = self.l1 * np.array([np.cos(t1), np.sin(t1)])
        tip = self._fingertip()
        lim = self.reach * 1.15
        fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=64)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.add_patch(plt.Circle((0, 0), self.reach, color="#e2e8f0", fill=False, ls="--", lw=1))
        ax.plot([0, elbow[0]], [0, elbow[1]], color="#334155", lw=6, solid_capstyle="round")
        ax.plot([elbow[0], tip[0]], [elbow[1], tip[1]], color="#2563eb", lw=6, solid_capstyle="round")
        ax.plot([0], [0], marker="o", color="#1e293b", ms=9)
        ax.plot(*self._target, marker="*", color="#f59e0b", ms=18)
        ax.plot(*tip, marker="o", color="#db2777", ms=7)
        frame = fig_to_rgb(fig)
        plt.close(fig)
        return frame
