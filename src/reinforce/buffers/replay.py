"""Uniform experience replay buffer for off-policy algorithms.

Supports multi-step (n-step) returns: set ``n_step > 1`` to store aggregated
n-step transitions. Each stored transition carries its own discount factor
(``gamma ** k``) so bootstrapping is exact even when an episode terminates or is
truncated inside the window.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from ..core.spaces import Space, is_discrete

__all__ = ["ReplayBuffer", "ReplayBatch"]


@dataclass
class ReplayBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor
    discounts: torch.Tensor  # per-sample discount for bootstrapping (gamma ** k)
    weights: Optional[torch.Tensor] = None  # importance weights (PER)
    indices: Optional[np.ndarray] = None  # sampled indices (PER)


def _shape(space: Space):
    return space.shape if space.shape is not None else ()


class ReplayBuffer:
    """A ring buffer of transitions with uniform random sampling.

    ``done`` stored here is the *bootstrapping* terminal flag (Gymnasium's
    ``terminated``), never a time-limit truncation, so target computation stays
    correct. Pass ``episode_end=True`` to :meth:`add` at any episode boundary
    (termination *or* truncation) so multi-step windows never bridge episodes.
    """

    def __init__(
        self,
        capacity: int,
        observation_space: Space,
        action_space: Space,
        device: str = "cpu",
        seed: Optional[int] = None,
        n_step: int = 1,
        gamma: float = 0.99,
    ) -> None:
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        self.n_step = int(n_step)
        self.gamma = float(gamma)

        self.discrete_actions = is_discrete(action_space)
        obs_shape = _shape(observation_space)
        obs_dtype = np.int64 if is_discrete(observation_space) else np.float32
        act_shape = () if self.discrete_actions else _shape(action_space)
        act_dtype = np.int64 if self.discrete_actions else np.float32

        self.obs = np.zeros((self.capacity, *obs_shape), dtype=obs_dtype)
        self.next_obs = np.zeros((self.capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.zeros((self.capacity, *act_shape), dtype=act_dtype)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)
        self.discounts = np.full((self.capacity,), self.gamma, dtype=np.float32)

        self.pos = 0
        self.size = 0
        self._nstep: deque = deque(maxlen=self.n_step)

    def __len__(self) -> int:
        return self.size

    # -- storage (overridden by PrioritizedReplayBuffer) -------------------
    def _store_transition(self, obs, action, reward, next_obs, done, discount) -> int:
        idx = self.pos
        self.obs[idx] = obs
        self.next_obs[idx] = next_obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = float(done)
        self.discounts[idx] = discount
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return idx

    def _emit_front(self) -> None:
        """Aggregate the current n-step window starting at its front and store it."""
        ret = 0.0
        next_o = self._nstep[-1][3]
        done = self._nstep[-1][4]
        k = len(self._nstep)
        for i, (_o, _a, r, no, d) in enumerate(self._nstep):
            ret += (self.gamma ** i) * r
            if d:
                next_o, done, k = no, True, i + 1
                break
        first_obs, first_action = self._nstep[0][0], self._nstep[0][1]
        self._store_transition(first_obs, first_action, ret, next_o, done, self.gamma ** k)
        self._nstep.popleft()

    def add(self, obs, action, reward, next_obs, done, episode_end: bool = False) -> None:
        self._nstep.append((np.asarray(obs), action, float(reward), np.asarray(next_obs), bool(done)))
        if len(self._nstep) >= self.n_step:
            self._emit_front()
        if done or episode_end:
            while self._nstep:  # flush the (shorter) trailing windows at episode end
                self._emit_front()

    def _to_batch(self, idx: np.ndarray) -> ReplayBatch:
        return ReplayBatch(
            obs=torch.as_tensor(self.obs[idx], device=self.device, dtype=torch.float32),
            actions=torch.as_tensor(
                self.actions[idx],
                device=self.device,
                dtype=torch.long if self.discrete_actions else torch.float32,
            ),
            rewards=torch.as_tensor(self.rewards[idx], device=self.device, dtype=torch.float32),
            next_obs=torch.as_tensor(self.next_obs[idx], device=self.device, dtype=torch.float32),
            dones=torch.as_tensor(self.dones[idx], device=self.device, dtype=torch.float32),
            discounts=torch.as_tensor(self.discounts[idx], device=self.device, dtype=torch.float32),
        )

    def sample(self, batch_size: int) -> ReplayBatch:
        idx = self.rng.integers(0, self.size, size=batch_size)
        return self._to_batch(idx)
