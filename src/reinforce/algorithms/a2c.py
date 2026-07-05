"""Advantage Actor-Critic (A2C), the synchronous variant of A3C (Mnih et al., 2016).

A single full-batch gradient step per rollout, using GAE advantages. Simpler and
faster per update than PPO; a strong baseline especially with several parallel
environments.
"""

from __future__ import annotations

from itertools import chain
from typing import Optional, Sequence

import torch

from ..core.env import Env
from ..utils.torch_utils import explained_variance
from .base import OnPolicyAgent

__all__ = ["A2C"]


class A2C(OnPolicyAgent):
    def __init__(
        self,
        env: Env,
        learning_rate: float = 7e-4,
        n_steps: int = 5,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        normalize_advantage: bool = False,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: str = "tanh",
        device: str = "auto",
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            env, n_steps=n_steps, gamma=gamma, gae_lambda=gae_lambda,
            learning_rate=learning_rate, hidden_sizes=hidden_sizes,
            activation=activation, device=device, seed=seed, **kwargs,
        )
        self.ent_coef = float(ent_coef)
        self.vf_coef = float(vf_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.normalize_advantage = bool(normalize_advantage)

    def _extra_config(self) -> dict:
        return dict(
            ent_coef=self.ent_coef, vf_coef=self.vf_coef,
            max_grad_norm=self.max_grad_norm, normalize_advantage=self.normalize_advantage,
        )

    def _update(self) -> dict:
        params = list(chain(self.actor.parameters(), self.critic.parameters()))
        # A2C performs a single gradient step over the whole rollout.
        batch = next(self.buffer.get(batch_size=None))
        log_probs, entropy, values = self._evaluate_actions(batch.obs, batch.actions)

        advantages = batch.advantages
        if self.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_loss = -(advantages * log_probs).mean()
        value_loss = ((batch.returns - values) ** 2).mean()
        entropy_loss = -entropy.mean()
        loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        self.optimizer.step()

        ev = explained_variance(self.buffer.values.ravel(), self.buffer.returns.ravel())
        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.mean().item()),
            "explained_variance": float(ev),
        }
