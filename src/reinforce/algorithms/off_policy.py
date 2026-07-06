"""Shared training loop for off-policy continuous-control agents (DDPG/TD3/SAC)."""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
import torch

from ..buffers.prioritized import PrioritizedReplayBuffer
from ..buffers.replay import ReplayBuffer
from ..core.agent import BaseAgent
from ..core.env import Env
from ..core.spaces import is_discrete
from ..exploration.schedules import LinearSchedule
from ..utils.torch_utils import get_device, to_tensor

__all__ = ["OffPolicyContinuousAgent"]


class OffPolicyContinuousAgent(BaseAgent):
    def __init__(
        self,
        env: Env,
        gamma: float = 0.99,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        learning_starts: int = 1_000,
        train_freq: int = 1,
        gradient_steps: int = 1,
        tau: float = 0.005,
        n_step: int = 1,
        prioritized: bool = False,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        device: str = "auto",
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(env, seed=seed, **kwargs)
        assert not is_discrete(self.action_space), "this agent requires a continuous (Box) action space"
        self.device = get_device(device)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.learning_starts = int(learning_starts)
        self.train_freq = int(train_freq)
        self.gradient_steps = int(gradient_steps)
        self.tau = float(tau)

        self.obs_dim = int(np.prod(self.observation_space.shape))
        self.act_dim = int(self.action_space.shape[0])
        self.action_low = np.asarray(self.action_space.low, dtype=np.float32)
        self.action_high = np.asarray(self.action_space.high, dtype=np.float32)

        self.n_step = int(n_step)
        self.prioritized = bool(prioritized)
        self.per_beta_start = float(per_beta_start)
        self._beta = float(per_beta_start)
        if self.prioritized:
            self.buffer = PrioritizedReplayBuffer(
                buffer_size, self.observation_space, self.action_space,
                alpha=per_alpha, beta=per_beta_start, device=str(self.device),
                seed=seed, n_step=self.n_step, gamma=self.gamma,
            )
        else:
            self.buffer = ReplayBuffer(
                buffer_size, self.observation_space, self.action_space,
                device=str(self.device), seed=seed, n_step=self.n_step, gamma=self.gamma,
            )

    # -- to be implemented by subclasses -----------------------------------
    def act(self, obs, deterministic: bool = False) -> np.ndarray:
        raise NotImplementedError

    def train_step(self) -> dict:
        raise NotImplementedError

    def _on_episode_end(self) -> None:
        """Hook, e.g. to reset correlated action noise."""

    def predict(self, obs, deterministic: bool = True) -> np.ndarray:
        return self.act(obs, deterministic=deterministic)

    def _tensor(self, obs) -> torch.Tensor:
        return to_tensor(np.asarray(obs, dtype=np.float32).reshape(1, -1), self.device)

    def _sample(self):
        """Sample a batch (prioritized with the current beta, or uniform)."""
        if self.prioritized:
            return self.buffer.sample(self.batch_size, beta=self._beta)
        return self.buffer.sample(self.batch_size)

    def _update_priorities(self, batch, td_errors: torch.Tensor) -> None:
        if self.prioritized and batch.indices is not None:
            self.buffer.update_priorities(batch.indices, td_errors.detach().cpu().numpy())

    def learn(self, total_steps: int, callback=None, log_interval: int = 10) -> "OffPolicyContinuousAgent":
        self._total_timesteps = self.num_timesteps + total_steps
        if callback is not None:
            callback.on_training_start(self)
        obs, _ = self.env.reset(seed=self.seed)
        ep_return, episodes = 0.0, 0
        returns_window: deque = deque(maxlen=100)
        metrics: dict = {}
        beta_schedule = LinearSchedule(self.per_beta_start, 1.0, total_steps)

        for step in range(total_steps):
            if step < self.learning_starts:
                action = self.action_space.sample()
            else:
                action = self.act(obs, deterministic=False)

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            self.buffer.add(obs, action, reward, next_obs, terminated,
                            episode_end=(terminated or truncated))
            obs = next_obs
            ep_return += reward
            self.num_timesteps += 1

            if step >= self.learning_starts and step % self.train_freq == 0:
                self._beta = beta_schedule(step)
                for _ in range(self.gradient_steps):
                    metrics = self.train_step()

            if callback is not None and not callback.on_step():
                break

            if terminated or truncated:
                episodes += 1
                returns_window.append(ep_return)
                self._on_episode_end()
                if episodes % log_interval == 0:
                    self.logger.record("rollout/ep_return_mean", float(np.mean(returns_window)))
                    for k, v in metrics.items():
                        self.logger.record(f"train/{k}", v)
                    self.logger.dump(self.num_timesteps)
                obs, _ = self.env.reset()
                ep_return = 0.0

        if callback is not None:
            callback.on_training_end()
        return self
