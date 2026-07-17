"""Admission control for a queue: an applied-RL operations / systems problem.

Jobs of varying value arrive at a finite-buffer server. Each step the agent
decides whether to admit the arriving job or reject it. Admitting a job captures
its value but occupies scarce buffer space and incurs a per-step holding cost
while it waits, and a full buffer blocks future (possibly higher-value) jobs. The
optimal policy is a value threshold that tightens as the queue fills -- admit
everything is the naive baseline. This is the RL version of load shedding / call
admission control.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..core.env import Env
from ..core.spaces import Box, Discrete

__all__ = ["QueueAdmissionControl"]


class QueueAdmissionControl(Env):
    def __init__(
        self,
        buffer_size: int = 10,
        service_prob: float = 0.5,
        holding_cost: float = 0.05,
        horizon: int = 100,
    ) -> None:
        self.buffer_size = int(buffer_size)
        self.service_prob = float(service_prob)
        self.holding_cost = float(holding_cost)
        self.horizon = int(horizon)

        # Observation: queue occupancy in [0, 1] and the incoming job's value in [0, 1].
        self.observation_space = Box(0.0, 1.0, shape=(2,), dtype=np.float32)
        self.action_space = Discrete(2)  # 0 = reject, 1 = admit

        self._rng = np.random.default_rng()
        self._queue = 0
        self._incoming = 0.0
        self._steps = 0

    def _obs(self) -> np.ndarray:
        return np.array([self._queue / self.buffer_size, self._incoming], dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._queue = 0
        self._incoming = float(self._rng.random())
        self._steps = 0
        return self._obs(), {}

    def step(self, action: int):
        # Server completes a job first (frees capacity).
        if self._queue > 0 and self._rng.random() < self.service_prob:
            self._queue -= 1

        reward = 0.0
        admitted = False
        if int(action) == 1 and self._queue < self.buffer_size:
            self._queue += 1
            reward += self._incoming  # capture the job's value
            admitted = True

        reward -= self.holding_cost * self._queue  # congestion cost

        self._incoming = float(self._rng.random())  # next arrival's value
        self._steps += 1
        truncated = self._steps >= self.horizon
        info = {"queue": self._queue, "admitted": admitted}
        return self._obs(), float(reward), False, truncated, info

    def render_rgb(self):
        from ..utils.render import bars_frame
        return bars_frame(["queue"], [self._queue], self.buffer_size,
                          title=f"incoming value {self._incoming:.2f}")
