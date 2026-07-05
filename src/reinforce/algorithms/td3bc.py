"""TD3+BC: minimalist offline RL (Fujimoto & Gu, 2021).

TD3 plus a behavior-cloning term in the actor loss, trained purely from a fixed
dataset with no environment interaction. Despite its simplicity it is a strong
offline-RL baseline. The BC weight is normalized by the average Q magnitude so a
single ``alpha`` works across tasks.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from ..core.env import Env
from ..data import TransitionDataset
from ..utils.torch_utils import get_device, soft_update
from .td3 import TD3

__all__ = ["TD3BC"]


class TD3BC(TD3):
    def __init__(self, env: Env, alpha: float = 2.5, **kwargs) -> None:
        super().__init__(env, **kwargs)
        self.alpha = float(alpha)

    def _offline_update(self, batch) -> dict:
        self._n_updates += 1
        with torch.no_grad():
            noise = (torch.randn_like(batch.actions) * self.policy_noise * self._noise_scale).clamp(
                -self.noise_clip * self._noise_scale, self.noise_clip * self._noise_scale
            )
            next_actions = (self.actor_target(batch.next_obs) + noise).clamp(
                self._act_low, self._act_high
            )
            target_q = torch.min(
                self.critic1_target(batch.next_obs, next_actions),
                self.critic2_target(batch.next_obs, next_actions),
            )
            y = batch.rewards + batch.discounts * (1.0 - batch.dones) * target_q

        q1 = self.critic1(batch.obs, batch.actions)
        q2 = self.critic2(batch.obs, batch.actions)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        metrics = {"critic_loss": float(critic_loss.item())}
        if self._n_updates % self.policy_delay == 0:
            pi = self.actor(batch.obs)
            q = self.critic1(batch.obs, pi)
            lmbda = self.alpha / q.abs().mean().detach()
            bc_loss = F.mse_loss(pi, batch.actions)
            actor_loss = -lmbda * q.mean() + bc_loss
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            soft_update(self.actor, self.actor_target, self.tau)
            soft_update(self.critic1, self.critic1_target, self.tau)
            soft_update(self.critic2, self.critic2_target, self.tau)
            metrics.update(actor_loss=float(actor_loss.item()), bc_loss=float(bc_loss.item()))
        return metrics

    def learn_offline(
        self,
        dataset: TransitionDataset,
        total_steps: int,
        callback=None,
        log_interval: int = 1000,
    ) -> "TD3BC":
        """Train from a fixed dataset (no environment interaction)."""
        self._total_timesteps = self.num_timesteps + total_steps
        if callback is not None:
            callback.on_training_start(self)
        losses: deque = deque(maxlen=100)
        for _ in range(total_steps):
            batch = dataset.sample(self.batch_size)
            metrics = self._offline_update(batch)
            losses.append(metrics["critic_loss"])
            self.num_timesteps += 1
            if callback is not None and not callback.on_step():
                break
            if self.num_timesteps % log_interval == 0:
                self.logger.record("train/critic_loss", float(np.mean(losses)))
                self.logger.dump(self.num_timesteps)
        if callback is not None:
            callback.on_training_end()
        return self

    # learn() from an env is not meaningful for an offline algorithm.
    def learn(self, *args, **kwargs):  # pragma: no cover - guard
        raise NotImplementedError("TD3BC is offline; use learn_offline(dataset, total_steps).")

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(),
                "config": dict(gamma=self.gamma, tau=self.tau, hidden_sizes=self.hidden_sizes,
                               policy_delay=self.policy_delay, alpha=self.alpha,
                               batch_size=self.batch_size, learning_starts=self.learning_starts),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, env: Env = None, device: str = "auto", **kwargs) -> "TD3BC":
        checkpoint = torch.load(path, map_location=get_device(device), weights_only=False)
        agent = cls(env, device=device, **{**checkpoint["config"], **kwargs})
        agent.actor.load_state_dict(checkpoint["actor"])
        agent.critic1.load_state_dict(checkpoint["critic1"])
        agent.critic2.load_state_dict(checkpoint["critic2"])
        from ..utils.torch_utils import hard_update

        hard_update(agent.critic1, agent.critic1_target)
        hard_update(agent.critic2, agent.critic2_target)
        hard_update(agent.actor, agent.actor_target)
        return agent
