"""Soft Actor-Critic (Haarnoja et al., 2018), with automatic entropy tuning.

Maximum-entropy off-policy RL: a stochastic squashed-Gaussian policy, twin
critics, and an automatically tuned temperature ``alpha`` that targets an
entropy of ``-|A|`` by default.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import torch

from ..core.env import Env
from ..networks.policies import SquashedGaussianActor
from ..networks.value import ContinuousQ
from ..utils.torch_utils import get_device, hard_update, soft_update
from .off_policy import OffPolicyContinuousAgent

__all__ = ["SAC"]


class SAC(OffPolicyContinuousAgent):
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
        ent_coef: Union[str, float] = "auto",
        target_entropy: Optional[float] = None,
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
        self.actor = SquashedGaussianActor(
            self.obs_dim, self.act_dim, self.action_low, self.action_high, self.hidden_sizes
        ).to(self.device)

        def make_critic():
            return ContinuousQ(self.obs_dim, self.act_dim, self.hidden_sizes).to(self.device)

        self.critic1, self.critic2 = make_critic(), make_critic()
        self.critic1_target, self.critic2_target = make_critic(), make_critic()
        hard_update(self.critic1, self.critic1_target)
        hard_update(self.critic2, self.critic2_target)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_opt = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=learning_rate
        )

        # Temperature (entropy coefficient).
        self.autotune = ent_coef == "auto"
        if self.autotune:
            self.target_entropy = (
                float(-self.act_dim) if target_entropy is None else float(target_entropy)
            )
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=learning_rate)
            self.alpha = float(self.log_alpha.exp().item())
        else:
            self.alpha = float(ent_coef)

    @torch.no_grad()
    def act(self, obs, deterministic: bool = False) -> np.ndarray:
        action, _, deterministic_action = self.actor.sample(self._tensor(obs))
        chosen = deterministic_action if deterministic else action
        return chosen.cpu().numpy()[0].astype(np.float32)

    def train_step(self) -> dict:
        batch = self._sample()

        with torch.no_grad():
            next_actions, next_log_prob, _ = self.actor.sample(batch.next_obs)
            next_log_prob = next_log_prob.squeeze(-1)
            target_q = torch.min(
                self.critic1_target(batch.next_obs, next_actions),
                self.critic2_target(batch.next_obs, next_actions),
            )
            target_q = target_q - self.alpha * next_log_prob
            y = batch.rewards + batch.discounts * (1.0 - batch.dones) * target_q

        q1 = self.critic1(batch.obs, batch.actions)
        q2 = self.critic2(batch.obs, batch.actions)
        td1, td2 = q1 - y, q2 - y
        weights = batch.weights if batch.weights is not None else torch.ones_like(td1)
        critic_loss = (weights * td1.pow(2)).mean() + (weights * td2.pow(2)).mean()
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        self._update_priorities(batch, 0.5 * (td1.abs() + td2.abs()))

        actions, log_prob, _ = self.actor.sample(batch.obs)
        log_prob = log_prob.squeeze(-1)
        q = torch.min(self.critic1(batch.obs, actions), self.critic2(batch.obs, actions))
        actor_loss = (self.alpha * log_prob - q).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        metrics = {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": self.alpha,
        }

        if self.autotune:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            self.alpha = float(self.log_alpha.exp().item())
            metrics["alpha_loss"] = float(alpha_loss.item())

        soft_update(self.critic1, self.critic1_target, self.tau)
        soft_update(self.critic2, self.critic2_target, self.tau)
        return metrics

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
    def load(cls, path: str, env: Env = None, device: str = "auto", **kwargs) -> "SAC":
        checkpoint = torch.load(path, map_location=get_device(device), weights_only=False)
        agent = cls(env, device=device, **{**checkpoint["config"], **kwargs})
        agent.actor.load_state_dict(checkpoint["actor"])
        agent.critic1.load_state_dict(checkpoint["critic1"])
        agent.critic2.load_state_dict(checkpoint["critic2"])
        hard_update(agent.critic1, agent.critic1_target)
        hard_update(agent.critic2, agent.critic2_target)
        return agent
