"""Twin Delayed DDPG (TD3, Fujimoto et al., 2018).

Three tricks over DDPG: (1) twin critics with a min to fight overestimation,
(2) delayed policy updates, (3) target-policy smoothing.
"""

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

__all__ = ["TD3"]


class TD3(OffPolicyContinuousAgent):
    def __init__(
        self,
        env: Env,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        learning_starts: int = 1_000,
        train_freq: int = 1,
        gradient_steps: int = 1,
        tau: float = 0.005,
        hidden_sizes: Sequence[int] = (256, 256),
        policy_delay: int = 2,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
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
        self.policy_delay = int(policy_delay)
        self.scale = (self.action_high - self.action_low) / 2.0
        # noise is defined in *scaled* action units
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self._n_updates = 0

        def make_actor():
            return DeterministicActor(
                self.obs_dim, self.act_dim, self.action_low, self.action_high, self.hidden_sizes
            ).to(self.device)

        def make_critic():
            return ContinuousQ(self.obs_dim, self.act_dim, self.hidden_sizes).to(self.device)

        self.actor = make_actor()
        self.actor_target = make_actor()
        hard_update(self.actor, self.actor_target)

        self.critic1, self.critic2 = make_critic(), make_critic()
        self.critic1_target, self.critic2_target = make_critic(), make_critic()
        hard_update(self.critic1, self.critic1_target)
        hard_update(self.critic2, self.critic2_target)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_opt = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=learning_rate
        )

        self._noise_scale = torch.as_tensor(self.scale, dtype=torch.float32, device=self.device)
        self._act_low = torch.as_tensor(self.action_low, device=self.device)
        self._act_high = torch.as_tensor(self.action_high, device=self.device)
        self.action_noise = action_noise or GaussianNoise(
            mean=np.zeros(self.act_dim), sigma=exploration_sigma * self.scale, seed=seed
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
            y = batch.rewards + self.gamma * (1.0 - batch.dones) * target_q

        q1 = self.critic1(batch.obs, batch.actions)
        q2 = self.critic2(batch.obs, batch.actions)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        metrics = {"critic_loss": float(critic_loss.item())}

        # Delayed policy & target updates.
        if self._n_updates % self.policy_delay == 0:
            actor_loss = -self.critic1(batch.obs, self.actor(batch.obs)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            soft_update(self.actor, self.actor_target, self.tau)
            soft_update(self.critic1, self.critic1_target, self.tau)
            soft_update(self.critic2, self.critic2_target, self.tau)
            metrics["actor_loss"] = float(actor_loss.item())
        return metrics

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(),
                "config": dict(
                    gamma=self.gamma, tau=self.tau, hidden_sizes=self.hidden_sizes,
                    policy_delay=self.policy_delay, batch_size=self.batch_size,
                    learning_starts=self.learning_starts,
                ),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, env: Env = None, device: str = "auto", **kwargs) -> "TD3":
        checkpoint = torch.load(path, map_location=get_device(device), weights_only=False)
        agent = cls(env, device=device, **{**checkpoint["config"], **kwargs})
        agent.actor.load_state_dict(checkpoint["actor"])
        agent.critic1.load_state_dict(checkpoint["critic1"])
        agent.critic2.load_state_dict(checkpoint["critic2"])
        hard_update(agent.actor, agent.actor_target)
        hard_update(agent.critic1, agent.critic1_target)
        hard_update(agent.critic2, agent.critic2_target)
        return agent
