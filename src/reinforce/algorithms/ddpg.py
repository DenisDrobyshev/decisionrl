"""Deep Deterministic Policy Gradient (Lillicrap et al., 2016)."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..core.env import Env
from ..exploration.noise import ActionNoise, GaussianNoise
from ..networks.policies import DeterministicActor
from ..networks.value import ContinuousQ
from ..utils.torch_utils import get_device, hard_update, soft_update
from .off_policy import OffPolicyContinuousAgent

__all__ = ["DDPG"]


class DDPG(OffPolicyContinuousAgent):
    def __init__(
        self,
        env: Env,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        learning_starts: int = 1_000,
        train_freq: int = 1,
        gradient_steps: int = 1,
        tau: float = 0.005,
        hidden_sizes: Sequence[int] = (256, 256),
        exploration_sigma: float = 0.1,
        action_noise: Optional[ActionNoise] = None,
        device: str = "auto",
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            env, gamma=gamma, buffer_size=buffer_size, batch_size=batch_size,
            learning_starts=learning_starts, train_freq=train_freq,
            gradient_steps=gradient_steps, tau=tau, device=device, seed=seed, **kwargs,
        )
        self.hidden_sizes = tuple(hidden_sizes)
        self.actor = DeterministicActor(
            self.obs_dim, self.act_dim, self.action_low, self.action_high, self.hidden_sizes
        ).to(self.device)
        self.actor_target = DeterministicActor(
            self.obs_dim, self.act_dim, self.action_low, self.action_high, self.hidden_sizes
        ).to(self.device)
        hard_update(self.actor, self.actor_target)

        self.critic = ContinuousQ(self.obs_dim, self.act_dim, self.hidden_sizes).to(self.device)
        self.critic_target = ContinuousQ(self.obs_dim, self.act_dim, self.hidden_sizes).to(self.device)
        hard_update(self.critic, self.critic_target)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)

        scale = (self.action_high - self.action_low) / 2.0
        self.action_noise = action_noise or GaussianNoise(
            mean=np.zeros(self.act_dim), sigma=exploration_sigma * scale, seed=seed
        )

    def _on_episode_end(self) -> None:
        self.action_noise.reset()

    @torch.no_grad()
    def act(self, obs, deterministic: bool = False) -> np.ndarray:
        action = self.actor(self._tensor(obs)).cpu().numpy()[0]
        if not deterministic:
            action = action + self.action_noise()
        return np.clip(action, self.action_low, self.action_high).astype(np.float32)

    def train_step(self) -> dict:
        batch = self.buffer.sample(self.batch_size)

        with torch.no_grad():
            next_actions = self.actor_target(batch.next_obs)
            target_q = self.critic_target(batch.next_obs, next_actions)
            y = batch.rewards + batch.discounts * (1.0 - batch.dones) * target_q

        current_q = self.critic(batch.obs, batch.actions)
        critic_loss = F.mse_loss(current_q, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        actor_loss = -self.critic(batch.obs, self.actor(batch.obs)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        soft_update(self.actor, self.actor_target, self.tau)
        soft_update(self.critic, self.critic_target, self.tau)
        return {"critic_loss": float(critic_loss.item()), "actor_loss": float(actor_loss.item())}

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "config": dict(gamma=self.gamma, tau=self.tau, hidden_sizes=self.hidden_sizes,
                               batch_size=self.batch_size, learning_starts=self.learning_starts),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, env: Env = None, device: str = "auto", **kwargs) -> "DDPG":
        checkpoint = torch.load(path, map_location=get_device(device), weights_only=False)
        agent = cls(env, device=device, **{**checkpoint["config"], **kwargs})
        agent.actor.load_state_dict(checkpoint["actor"])
        agent.critic.load_state_dict(checkpoint["critic"])
        hard_update(agent.actor, agent.actor_target)
        hard_update(agent.critic, agent.critic_target)
        return agent
