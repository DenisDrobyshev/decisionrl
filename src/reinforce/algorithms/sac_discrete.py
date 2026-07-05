"""Discrete SAC: Soft Actor-Critic for discrete action spaces (Christodoulou, 2019).

A categorical policy with twin Q-networks that output a value for every action.
Because the action set is finite, the soft value and the policy/temperature
losses are computed as exact expectations over actions (no sampling). Automatic
entropy tuning targets a fraction of the maximum entropy ``log|A|``.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..buffers.replay import ReplayBuffer
from ..core.agent import BaseAgent
from ..core.env import Env
from ..core.spaces import is_discrete
from ..networks.policies import CategoricalActor
from ..networks.q_networks import QNetwork
from ..utils.torch_utils import get_device, hard_update, soft_update, to_tensor

__all__ = ["SACDiscrete"]


class SACDiscrete(BaseAgent):
    def __init__(
        self,
        env: Env,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        learning_starts: int = 1_000,
        train_freq: int = 1,
        gradient_steps: int = 1,
        tau: float = 0.005,
        hidden_sizes: Sequence[int] = (128, 128),
        ent_coef: Union[str, float] = "auto",
        target_entropy_ratio: float = 0.98,
        n_step: int = 1,
        device: str = "auto",
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(env, seed=seed, **kwargs)
        assert is_discrete(self.action_space), "SACDiscrete requires a discrete action space"
        assert self.observation_space.shape is not None, "SACDiscrete requires a Box observation space"

        self.device = get_device(device)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.learning_starts = int(learning_starts)
        self.train_freq = int(train_freq)
        self.gradient_steps = int(gradient_steps)
        self.tau = float(tau)
        self.hidden_sizes = tuple(hidden_sizes)
        self.n_step = int(n_step)

        self.obs_dim = int(np.prod(self.observation_space.shape))
        self.n_actions = int(self.action_space.n)

        self.actor = CategoricalActor(self.obs_dim, self.n_actions, self.hidden_sizes, nn.ReLU).to(self.device)

        def make_critic():
            return QNetwork(self.obs_dim, self.n_actions, self.hidden_sizes, nn.ReLU).to(self.device)

        self.critic1, self.critic2 = make_critic(), make_critic()
        self.critic1_target, self.critic2_target = make_critic(), make_critic()
        hard_update(self.critic1, self.critic1_target)
        hard_update(self.critic2, self.critic2_target)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_opt = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=learning_rate
        )

        self.autotune = ent_coef == "auto"
        if self.autotune:
            self.target_entropy = float(target_entropy_ratio) * math.log(self.n_actions)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=learning_rate)
            self.alpha = float(self.log_alpha.exp().item())
        else:
            self.alpha = float(ent_coef)

        self.buffer = ReplayBuffer(
            buffer_size, self.observation_space, self.action_space,
            device=str(self.device), seed=seed, n_step=self.n_step, gamma=self.gamma,
        )

    @torch.no_grad()
    def predict(self, obs, deterministic: bool = True) -> int:
        obs_t = to_tensor(np.asarray(obs).reshape(1, -1), self.device)
        dist = self.actor(obs_t)
        action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        return int(action.item())

    def _probs(self, obs):
        dist = self.actor(obs)
        probs = dist.probs
        log_probs = torch.log(probs + 1e-8)
        return probs, log_probs

    def train_step(self) -> dict:
        batch = self.buffer.sample(self.batch_size)
        actions = batch.actions.view(-1, 1)

        with torch.no_grad():
            next_probs, next_log_probs = self._probs(batch.next_obs)
            next_q = torch.min(self.critic1_target(batch.next_obs), self.critic2_target(batch.next_obs))
            next_v = (next_probs * (next_q - self.alpha * next_log_probs)).sum(dim=1)
            target = batch.rewards + batch.discounts * (1.0 - batch.dones) * next_v

        q1 = self.critic1(batch.obs).gather(1, actions).squeeze(1)
        q2 = self.critic2(batch.obs).gather(1, actions).squeeze(1)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        probs, log_probs = self._probs(batch.obs)
        with torch.no_grad():
            q = torch.min(self.critic1(batch.obs), self.critic2(batch.obs))
        actor_loss = (probs * (self.alpha * log_probs - q)).sum(dim=1).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        metrics = {"critic_loss": float(critic_loss.item()), "actor_loss": float(actor_loss.item()), "alpha": self.alpha}
        if self.autotune:
            alpha_loss = -(
                self.log_alpha * (log_probs + self.target_entropy).detach() * probs.detach()
            ).sum(dim=1).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            self.alpha = float(self.log_alpha.exp().item())
            metrics["alpha_loss"] = float(alpha_loss.item())

        soft_update(self.critic1, self.critic1_target, self.tau)
        soft_update(self.critic2, self.critic2_target, self.tau)
        return metrics

    def learn(self, total_steps: int, callback=None, log_interval: int = 10) -> "SACDiscrete":
        if callback is not None:
            callback.on_training_start(self)
        obs, _ = self.env.reset(seed=self.seed)
        ep_return, episodes = 0.0, 0
        returns_window: deque = deque(maxlen=100)
        metrics: dict = {}

        for step in range(total_steps):
            if step < self.learning_starts:
                action = int(self.rng.integers(self.n_actions))
            else:
                action = self.predict(obs, deterministic=False)

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            self.buffer.add(obs, action, reward, next_obs, terminated, episode_end=(terminated or truncated))
            obs = next_obs
            ep_return += reward
            self.num_timesteps += 1

            if step >= self.learning_starts and step % self.train_freq == 0:
                for _ in range(self.gradient_steps):
                    metrics = self.train_step()

            if callback is not None and not callback.on_step():
                break

            if terminated or truncated:
                episodes += 1
                returns_window.append(ep_return)
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

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(),
                "config": dict(
                    gamma=self.gamma, tau=self.tau, hidden_sizes=self.hidden_sizes,
                    batch_size=self.batch_size, learning_starts=self.learning_starts,
                    ent_coef="auto" if self.autotune else self.alpha,
                ),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, env: Env = None, device: str = "auto", **kwargs) -> "SACDiscrete":
        checkpoint = torch.load(path, map_location=get_device(device), weights_only=False)
        agent = cls(env, device=device, **{**checkpoint["config"], **kwargs})
        agent.actor.load_state_dict(checkpoint["actor"])
        agent.critic1.load_state_dict(checkpoint["critic1"])
        agent.critic2.load_state_dict(checkpoint["critic2"])
        hard_update(agent.critic1, agent.critic1_target)
        hard_update(agent.critic2, agent.critic2_target)
        return agent
