"""Proximal Policy Optimization (Schulman et al., 2017).

Includes the practical details that make PPO work: GAE, advantage
normalization, a clipped surrogate objective, optional value-function clipping,
entropy bonus, gradient clipping and early stopping on a target KL divergence.
"""

from __future__ import annotations

from itertools import chain
from typing import Optional, Sequence

import numpy as np
import torch

from ..core.env import Env
from ..utils.torch_utils import explained_variance
from .base import OnPolicyAgent

__all__ = ["PPO"]


class PPO(OnPolicyAgent):
    def __init__(
        self,
        env: Env,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        clip_range_vf: Optional[float] = None,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        normalize_advantage: bool = True,
        target_kl: Optional[float] = None,
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
        self.batch_size = int(batch_size)
        self.n_epochs = int(n_epochs)
        self.clip_range = float(clip_range)
        self.clip_range_vf = clip_range_vf
        self.ent_coef = float(ent_coef)
        self.vf_coef = float(vf_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.normalize_advantage = bool(normalize_advantage)
        self.target_kl = target_kl

    def _extra_config(self) -> dict:
        return dict(
            batch_size=self.batch_size, n_epochs=self.n_epochs, clip_range=self.clip_range,
            clip_range_vf=self.clip_range_vf, ent_coef=self.ent_coef, vf_coef=self.vf_coef,
            max_grad_norm=self.max_grad_norm, normalize_advantage=self.normalize_advantage,
            target_kl=self.target_kl,
        )

    def _update(self) -> dict:
        params = list(chain(self.actor.parameters(), self.critic.parameters()))
        pg_losses, vf_losses, entropies, kls, clip_fractions = [], [], [], [], []

        continue_training = True
        for _epoch in range(self.n_epochs):
            if not continue_training:
                break
            for batch in self.buffer.get(self.batch_size):
                log_probs, entropy, values = self._evaluate_actions(batch.obs, batch.actions)

                advantages = batch.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = torch.exp(log_probs - batch.old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                if self.clip_range_vf is not None:
                    values_clipped = batch.old_values + torch.clamp(
                        values - batch.old_values, -self.clip_range_vf, self.clip_range_vf
                    )
                    vf_loss = torch.max(
                        (values - batch.returns) ** 2, (values_clipped - batch.returns) ** 2
                    ).mean()
                else:
                    vf_loss = ((values - batch.returns) ** 2).mean()

                entropy_loss = -entropy.mean()
                loss = policy_loss + self.vf_coef * vf_loss + self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_range).float().mean().item()
                pg_losses.append(policy_loss.item())
                vf_losses.append(vf_loss.item())
                entropies.append(entropy.mean().item())
                kls.append(approx_kl)
                clip_fractions.append(clip_fraction)

                if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:
                    continue_training = False
                    break

        ev = explained_variance(self.buffer.values.ravel(), self.buffer.returns.ravel())
        return {
            "policy_loss": float(np.mean(pg_losses)),
            "value_loss": float(np.mean(vf_losses)),
            "entropy": float(np.mean(entropies)),
            "approx_kl": float(np.mean(kls)),
            "clip_fraction": float(np.mean(clip_fractions)),
            "explained_variance": float(ev),
        }
