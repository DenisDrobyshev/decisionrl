"""Offline datasets of transitions for offline / batch RL.

A :class:`TransitionDataset` is a fixed set of ``(obs, action, reward, next_obs,
done)`` tuples that offline algorithms (e.g. :class:`~reinforce.algorithms.TD3BC`)
learn from without any further environment interaction. :func:`collect_dataset`
records one by rolling out a behaviour policy.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
import torch

from .buffers.replay import ReplayBatch
from .core.env import Env
from .core.spaces import is_discrete

__all__ = [
    "TransitionDataset",
    "collect_dataset",
    "TrajectoryDataset",
    "collect_trajectories",
]


class TransitionDataset:
    def __init__(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
        gamma: float = 0.99,
        device: str = "cpu",
        seed: Optional[int] = None,
    ) -> None:
        self.obs = np.asarray(obs, dtype=np.float32)
        self.actions = np.asarray(actions, dtype=np.float32)
        self.rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
        self.next_obs = np.asarray(next_obs, dtype=np.float32)
        self.dones = np.asarray(dones, dtype=np.float32).reshape(-1)
        self.gamma = float(gamma)
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        assert len(self.obs) == len(self.actions) == len(self.rewards)

    def __len__(self) -> int:
        return len(self.obs)

    def sample(self, batch_size: int) -> ReplayBatch:
        idx = self.rng.integers(0, len(self), size=batch_size)
        t = lambda a, d=torch.float32: torch.as_tensor(a[idx], device=self.device, dtype=d)  # noqa: E731
        return ReplayBatch(
            obs=t(self.obs),
            actions=t(self.actions),
            rewards=t(self.rewards),
            next_obs=t(self.next_obs),
            dones=t(self.dones),
            discounts=torch.full((batch_size,), self.gamma, device=self.device),
        )

    def normalize_obs(self, eps: float = 1e-3):
        """Return ``(mean, std)`` of observations and standardize in place."""
        mean = self.obs.mean(axis=0)
        std = self.obs.std(axis=0) + eps
        self.obs = (self.obs - mean) / std
        self.next_obs = (self.next_obs - mean) / std
        return mean, std


def collect_dataset(
    env: Env,
    policy: Callable[[np.ndarray], np.ndarray],
    n_transitions: int,
    gamma: float = 0.99,
    seed: Optional[int] = None,
    device: str = "cpu",
) -> TransitionDataset:
    """Roll out ``policy`` in ``env`` and record ``n_transitions`` transitions."""
    obs_buf, act_buf, rew_buf, next_buf, done_buf = [], [], [], [], []
    obs, _ = env.reset(seed=seed)
    for _ in range(n_transitions):
        action = policy(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        obs_buf.append(np.asarray(obs, dtype=np.float32))
        act_buf.append(np.asarray(action, dtype=np.float32))
        rew_buf.append(reward)
        next_buf.append(np.asarray(next_obs, dtype=np.float32))
        done_buf.append(terminated)
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
    return TransitionDataset(
        np.array(obs_buf), np.array(act_buf), np.array(rew_buf),
        np.array(next_buf), np.array(done_buf), gamma=gamma, device=device, seed=seed,
    )


class TrajectoryDataset:
    """Full episodes with precomputed returns-to-go, for sequence models.

    Unlike :class:`TransitionDataset` (unordered transitions), this keeps whole
    trajectories intact and serves fixed-length sub-sequences with the
    return-to-go at each step — exactly what a
    :class:`~reinforce.algorithms.DecisionTransformer` conditions on.
    """

    def __init__(
        self,
        trajectories: List[dict],
        discrete: bool,
        device: str = "cpu",
        seed: Optional[int] = None,
    ) -> None:
        self.discrete = bool(discrete)
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        self.trajectories = []
        for traj in trajectories:
            obs = np.asarray(traj["obs"], dtype=np.float32)
            rewards = np.asarray(traj["rewards"], dtype=np.float32).reshape(-1)
            actions = np.asarray(traj["actions"])
            actions = actions.astype(np.int64) if self.discrete else actions.astype(np.float32)
            # Return-to-go: reverse cumulative sum of future rewards.
            rtg = np.flip(np.cumsum(np.flip(rewards))).copy().astype(np.float32)
            self.trajectories.append(
                {"obs": obs, "actions": actions, "rewards": rewards, "rtg": rtg}
            )
        self.obs_dim = int(self.trajectories[0]["obs"].shape[1])
        if self.discrete:
            self.act_dim = 1
        else:
            self.act_dim = int(self.trajectories[0]["actions"].shape[1])
        self.returns = np.array([float(t["rtg"][0]) for t in self.trajectories], dtype=np.float32)
        self.max_length = max(len(t["obs"]) for t in self.trajectories)

    def __len__(self) -> int:
        return len(self.trajectories)

    def sample(self, batch_size: int, context_len: int = 20, max_ep_len: int = 1000):
        """Return ``(states, actions, rtg, timesteps, mask)`` left-padded to ``K``."""
        K = int(context_len)
        states = np.zeros((batch_size, K, self.obs_dim), dtype=np.float32)
        if self.discrete:
            actions = np.zeros((batch_size, K), dtype=np.int64)
        else:
            actions = np.zeros((batch_size, K, self.act_dim), dtype=np.float32)
        rtg = np.zeros((batch_size, K, 1), dtype=np.float32)
        timesteps = np.zeros((batch_size, K), dtype=np.int64)
        mask = np.zeros((batch_size, K), dtype=np.float32)

        for b in range(batch_size):
            traj = self.trajectories[self.rng.integers(0, len(self.trajectories))]
            length = len(traj["obs"])
            si = int(self.rng.integers(0, length))
            end = min(si + K, length)
            L = end - si
            states[b, K - L :] = traj["obs"][si:end]
            actions[b, K - L :] = traj["actions"][si:end]
            rtg[b, K - L :, 0] = traj["rtg"][si:end]
            timesteps[b, K - L :] = np.clip(np.arange(si, end), 0, max_ep_len - 1)
            mask[b, K - L :] = 1.0

        t = lambda a, d=torch.float32: torch.as_tensor(a, device=self.device, dtype=d)  # noqa: E731
        return (
            t(states),
            t(actions, torch.long) if self.discrete else t(actions),
            t(rtg),
            t(timesteps, torch.long),
            t(mask),
        )


def collect_trajectories(
    env: Env,
    policy: Callable[[np.ndarray], object],
    n_trajectories: int,
    seed: Optional[int] = None,
    device: str = "cpu",
) -> TrajectoryDataset:
    """Roll out ``policy`` and record ``n_trajectories`` complete episodes."""
    trajectories = []
    for i in range(n_trajectories):
        ep_seed = None if seed is None else seed + i
        obs, _ = env.reset(seed=ep_seed)
        obs_l, act_l, rew_l = [], [], []
        done = False
        while not done:
            action = policy(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            obs_l.append(np.asarray(obs, dtype=np.float32))
            act_l.append(action)
            rew_l.append(float(reward))
            obs = next_obs
            done = terminated or truncated
        trajectories.append(
            {"obs": np.asarray(obs_l), "actions": np.asarray(act_l), "rewards": np.asarray(rew_l)}
        )
    return TrajectoryDataset(
        trajectories, discrete=is_discrete(env.action_space), device=device, seed=seed
    )
