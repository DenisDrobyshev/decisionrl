"""IMPALA-style V-trace actor-critic (Espeholt et al., 2018).

Collects rollouts from parallel actors (use a ``SyncVectorEnv`` / ``AsyncVectorEnv``)
and corrects for the policy lag between the behaviour policy (that generated the
data) and the current target policy with V-trace off-policy importance weighting.
Running several update epochs per rollout makes the behaviour/target policies
diverge, which is exactly what V-trace is designed to handle. Discrete and
continuous action spaces are supported (reuses the on-policy machinery).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch

from ..core.env import Env
from ..utils.torch_utils import explained_variance, to_tensor
from .base import OnPolicyAgent

__all__ = ["IMPALA"]


class IMPALA(OnPolicyAgent):
    def __init__(
        self,
        env: Env,
        learning_rate: float = 3e-4,
        n_steps: int = 32,
        n_epochs: int = 1,
        gamma: float = 0.99,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rho_bar: float = 1.0,
        c_bar: float = 1.0,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: str = "tanh",
        anneal_lr: bool = False,
        device: str = "auto",
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            env, n_steps=n_steps, gamma=gamma, gae_lambda=1.0, learning_rate=learning_rate,
            hidden_sizes=hidden_sizes, activation=activation, anneal_lr=anneal_lr,
            device=device, seed=seed, **kwargs,
        )
        self.n_epochs = int(n_epochs)
        self.ent_coef = float(ent_coef)
        self.vf_coef = float(vf_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.rho_bar = float(rho_bar)
        self.c_bar = float(c_bar)

    def _extra_config(self) -> dict:
        return dict(n_epochs=self.n_epochs, ent_coef=self.ent_coef, vf_coef=self.vf_coef,
                    max_grad_norm=self.max_grad_norm, rho_bar=self.rho_bar, c_bar=self.c_bar)

    def _update(self) -> dict:
        T, B = self.n_steps, self.num_envs
        buf = self.buffer
        obs = to_tensor(buf.obs.reshape(T * B, -1), self.device)
        if self.discrete:
            actions = torch.as_tensor(buf.actions.reshape(T * B), device=self.device, dtype=torch.long)
        else:
            actions = to_tensor(buf.actions.reshape(T * B, self.act_dim), self.device)
        behavior_logp = to_tensor(buf.log_probs, self.device)  # (T, B)
        rewards = to_tensor(buf.rewards, self.device)
        episode_starts = to_tensor(buf.episode_starts, self.device)
        with torch.no_grad():
            bootstrap = self.critic(to_tensor(np.asarray(self._last_obs, np.float32), self.device))
        last_done = to_tensor(self._last_episode_starts, self.device)

        params = list(self.actor.parameters()) + list(self.critic.parameters())
        metrics: dict = {}
        for _ in range(self.n_epochs):
            log_probs, entropy, values = self._evaluate_actions(obs, actions)
            log_probs = log_probs.reshape(T, B)
            entropy = entropy.reshape(T, B)
            values = values.reshape(T, B)

            with torch.no_grad():
                rho = torch.exp(log_probs - behavior_logp)
                clipped_rho = rho.clamp(max=self.rho_bar)
                c = rho.clamp(max=self.c_bar)
                v = values.detach()
                vs = torch.zeros_like(v)
                pg_adv = torch.zeros_like(v)
                carry = torch.zeros(B, device=self.device)
                for t in reversed(range(T)):
                    if t == T - 1:
                        nnt = 1.0 - last_done
                        next_v, next_vs = bootstrap, bootstrap
                    else:
                        nnt = 1.0 - episode_starts[t + 1]
                        next_v, next_vs = v[t + 1], vs[t + 1]
                    delta = clipped_rho[t] * (rewards[t] + self.gamma * nnt * next_v - v[t])
                    carry = delta + self.gamma * nnt * c[t] * carry
                    vs[t] = v[t] + carry
                    pg_adv[t] = clipped_rho[t] * (rewards[t] + self.gamma * nnt * next_vs - v[t])

            policy_loss = -(pg_adv * log_probs).mean()
            value_loss = ((vs - values) ** 2).mean()
            entropy_loss = -entropy.mean()
            loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
            self.optimizer.step()
            metrics = {
                "policy_loss": float(policy_loss.item()),
                "value_loss": float(value_loss.item()),
                "entropy": float(entropy.mean().item()),
                "mean_rho": float(rho.mean().item()),
            }
        metrics["explained_variance"] = float(
            explained_variance(self.buffer.values.ravel(), vs.cpu().numpy().ravel())
        )
        return metrics
