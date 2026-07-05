"""Deep Q-Network with Double, Dueling and Prioritized-Replay options.

References: Mnih et al. (2015, DQN), van Hasselt et al. (2016, Double DQN),
Wang et al. (2016, Dueling), Schaul et al. (2016, PER). All improvements are
toggleable so the agent doubles as a study of what each contributes.
"""

from __future__ import annotations

from collections import deque
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..buffers.prioritized import PrioritizedReplayBuffer
from ..buffers.replay import ReplayBuffer
from ..core.agent import BaseAgent
from ..core.env import Env
from ..core.spaces import is_discrete
from ..exploration.schedules import LinearSchedule
from ..networks.cnn import ImageQNetwork, is_image_space
from ..networks.q_networks import DuelingQNetwork, QNetwork
from ..utils.torch_utils import get_device, polyak_update, to_tensor

__all__ = ["DQN"]


class DQN(BaseAgent):
    def __init__(
        self,
        env: Env,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        learning_starts: int = 1_000,
        train_freq: int = 1,
        gradient_steps: int = 1,
        target_update_interval: int = 500,
        tau: float = 1.0,
        hidden_sizes: Sequence[int] = (128, 128),
        features_dim: int = 256,
        exploration_fraction: float = 0.1,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        max_grad_norm: float = 10.0,
        double_q: bool = True,
        dueling: bool = False,
        prioritized: bool = False,
        n_step: int = 1,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        device: str = "auto",
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(env, seed=seed, **kwargs)
        assert is_discrete(self.action_space), "DQN requires a discrete action space"
        assert self.observation_space.shape is not None, (
            "DQN requires a Box observation space (use one-hot encoding for discrete states)"
        )

        self.device = get_device(device)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.learning_starts = int(learning_starts)
        self.train_freq = int(train_freq)
        self.gradient_steps = int(gradient_steps)
        self.target_update_interval = int(target_update_interval)
        self.tau = float(tau)
        self.max_grad_norm = float(max_grad_norm)
        self.double_q = bool(double_q)
        self.dueling = bool(dueling)
        self.prioritized = bool(prioritized)
        self.exploration_fraction = float(exploration_fraction)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.hidden_sizes = tuple(hidden_sizes)
        self.per_beta_start = float(per_beta_start)
        self.n_step = int(n_step)

        self.obs_shape = tuple(self.observation_space.shape)
        self.is_image = is_image_space(self.observation_space)
        self.obs_dim = int(np.prod(self.observation_space.shape))
        self.n_actions = int(self.action_space.n)
        self.features_dim = int(features_dim)

        self._build_networks(learning_rate)

        if prioritized:
            self.buffer = PrioritizedReplayBuffer(
                buffer_size, self.observation_space, self.action_space,
                alpha=per_alpha, beta=per_beta_start, device=str(self.device), seed=seed,
                n_step=self.n_step, gamma=self.gamma,
            )
        else:
            self.buffer = ReplayBuffer(
                buffer_size, self.observation_space, self.action_space,
                device=str(self.device), seed=seed, n_step=self.n_step, gamma=self.gamma,
            )
        self.epsilon = epsilon_start

    def _build_networks(self, learning_rate: float) -> None:
        if self.is_image:
            def make():
                return ImageQNetwork(self.obs_shape, self.n_actions, self.features_dim)
        else:
            net_cls = DuelingQNetwork if self.dueling else QNetwork

            def make():
                return net_cls(self.obs_dim, self.n_actions, self.hidden_sizes)

        self.q_net = make().to(self.device)
        self.target_net = make().to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)

    @torch.no_grad()
    def predict(self, obs, deterministic: bool = True) -> int:
        if not deterministic and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        obs_t = to_tensor(np.asarray(obs).reshape(1, *self.obs_shape), self.device)
        return int(self.q_net(obs_t).argmax(dim=1).item())

    def _train_step(self, beta: float) -> float:
        if self.prioritized:
            batch = self.buffer.sample(self.batch_size, beta=beta)
        else:
            batch = self.buffer.sample(self.batch_size)

        with torch.no_grad():
            if self.double_q:
                next_actions = self.q_net(batch.next_obs).argmax(dim=1, keepdim=True)
                next_q = self.target_net(batch.next_obs).gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target_net(batch.next_obs).max(dim=1).values
            target = batch.rewards + batch.discounts * (1.0 - batch.dones) * next_q

        current_q = self.q_net(batch.obs).gather(1, batch.actions.view(-1, 1)).squeeze(1)
        td_error = current_q - target

        if self.prioritized:
            loss = (batch.weights * F.smooth_l1_loss(current_q, target, reduction="none")).mean()
        else:
            loss = F.smooth_l1_loss(current_q, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        if self.prioritized:
            self.buffer.update_priorities(batch.indices, td_error.detach().cpu().numpy())
        return float(loss.item())

    def learn(self, total_steps: int, callback=None, log_interval: int = 10) -> "DQN":
        eps_schedule = LinearSchedule(
            self.epsilon_start, self.epsilon_end,
            max(1, int(self.exploration_fraction * total_steps)),
        )
        beta_schedule = LinearSchedule(self.per_beta_start, 1.0, total_steps)
        self._total_timesteps = self.num_timesteps + total_steps
        if callback is not None:
            callback.on_training_start(self)

        obs, _ = self.env.reset(seed=self.seed)
        ep_return, ep_len, episodes = 0.0, 0, 0
        returns_window: deque = deque(maxlen=100)
        losses: deque = deque(maxlen=100)

        for step in range(total_steps):
            self.epsilon = eps_schedule(step)
            if step < self.learning_starts:
                action = int(self.rng.integers(self.n_actions))
            else:
                action = self.predict(obs, deterministic=False)

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            self.buffer.add(obs, action, reward, next_obs, terminated,
                            episode_end=(terminated or truncated))
            obs = next_obs
            ep_return += reward
            ep_len += 1
            self.num_timesteps += 1

            if step >= self.learning_starts and step % self.train_freq == 0:
                for _ in range(self.gradient_steps):
                    losses.append(self._train_step(beta_schedule(step)))

            if step % self.target_update_interval == 0:
                polyak_update(self.q_net, self.target_net, self.tau)

            if callback is not None and not callback.on_step():
                break

            if terminated or truncated:
                episodes += 1
                returns_window.append(ep_return)
                if episodes % log_interval == 0:
                    self.logger.record("rollout/ep_return_mean", float(np.mean(returns_window)))
                    self.logger.record("rollout/epsilon", self.epsilon)
                    if losses:
                        self.logger.record("train/loss", float(np.mean(losses)))
                    self.logger.dump(self.num_timesteps)
                obs, _ = self.env.reset()
                ep_return, ep_len = 0.0, 0

        if callback is not None:
            callback.on_training_end()
        return self

    # -- persistence -------------------------------------------------------
    def _config(self) -> dict:
        return dict(
            gamma=self.gamma, batch_size=self.batch_size, learning_starts=self.learning_starts,
            train_freq=self.train_freq, gradient_steps=self.gradient_steps,
            target_update_interval=self.target_update_interval, tau=self.tau,
            hidden_sizes=self.hidden_sizes, features_dim=self.features_dim,
            double_q=self.double_q, dueling=self.dueling,
            prioritized=self.prioritized, n_step=self.n_step, max_grad_norm=self.max_grad_norm,
        )

    def save(self, path: str) -> None:
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self._config(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, env: Optional[Env] = None, device: str = "auto", **kwargs) -> "DQN":
        checkpoint = torch.load(path, map_location=get_device(device), weights_only=False)
        agent = cls(env, device=device, **{**checkpoint["config"], **kwargs})
        agent.q_net.load_state_dict(checkpoint["q_net"])
        agent.target_net.load_state_dict(checkpoint["q_net"])
        agent.optimizer.load_state_dict(checkpoint["optimizer"])
        return agent
