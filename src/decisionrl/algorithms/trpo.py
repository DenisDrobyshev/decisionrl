"""Trust Region Policy Optimization (Schulman et al., 2015).

Takes the largest policy step that keeps the KL divergence from the old policy
within a trust region ``max_kl``. The natural-gradient direction is found with
conjugate gradient on the Fisher-vector product (the KL Hessian), then a
backtracking line search enforces the KL constraint and a surrogate improvement.
The value function is fit separately by regression. Built on the shared on-policy
rollout machinery (GAE, correct time-limit bootstrapping).
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch
from torch.distributions import kl_divergence

from ..core.env import Env
from .base import OnPolicyAgent

__all__ = ["TRPO"]


def _flat(vectors) -> torch.Tensor:
    return torch.cat([v.reshape(-1) for v in vectors])


def _flat_grad(output, params, retain_graph=False, create_graph=False) -> torch.Tensor:
    grads = torch.autograd.grad(output, list(params), retain_graph=retain_graph or create_graph,
                                create_graph=create_graph, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                      for g, p in zip(grads, params)])


class TRPO(OnPolicyAgent):
    def __init__(
        self,
        env: Env,
        n_steps: int = 2048,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        max_kl: float = 0.01,
        cg_iters: int = 10,
        cg_damping: float = 0.1,
        backtrack_iters: int = 10,
        backtrack_coeff: float = 0.8,
        vf_iters: int = 5,
        vf_lr: float = 1e-3,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: str = "tanh",
        device: str = "auto",
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(env, n_steps=n_steps, gamma=gamma, gae_lambda=gae_lambda,
                         hidden_sizes=hidden_sizes, activation=activation, device=device,
                         seed=seed, **kwargs)
        self.max_kl = float(max_kl)
        self.cg_iters = int(cg_iters)
        self.cg_damping = float(cg_damping)
        self.backtrack_iters = int(backtrack_iters)
        self.backtrack_coeff = float(backtrack_coeff)
        self.vf_iters = int(vf_iters)
        self.vf_opt = torch.optim.Adam(self.critic.parameters(), lr=vf_lr)

    def _extra_config(self) -> dict:
        return dict(max_kl=self.max_kl, cg_iters=self.cg_iters, cg_damping=self.cg_damping,
                    vf_iters=self.vf_iters)

    def _actor_params(self):
        return list(self.actor.parameters())

    def _get_flat_params(self) -> torch.Tensor:
        return _flat([p.data for p in self._actor_params()])

    def _set_flat_params(self, flat: torch.Tensor) -> None:
        i = 0
        for p in self._actor_params():
            n = p.numel()
            p.data.copy_(flat[i:i + n].view_as(p))
            i += n

    def _log_probs(self, dist, actions):
        return dist.log_prob(actions) if self.discrete else dist.log_prob(actions).sum(dim=-1)

    def _conjugate_gradient(self, fvp: Callable, b: torch.Tensor) -> torch.Tensor:
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rr = torch.dot(r, r)
        for _ in range(self.cg_iters):
            Ap = fvp(p)
            alpha = rr / (torch.dot(p, Ap) + 1e-8)
            x += alpha * p
            r -= alpha * Ap
            new_rr = torch.dot(r, r)
            if new_rr < 1e-10:
                break
            p = r + (new_rr / rr) * p
            rr = new_rr
        return x

    def _update(self) -> dict:
        batch = next(iter(self.buffer.get(self.n_steps * self.num_envs)))
        obs, actions, old_log_probs = batch.obs, batch.actions, batch.old_log_probs
        advantages = batch.advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        with torch.no_grad():
            old_dist = self._distribution(obs)

        def surrogate() -> torch.Tensor:
            dist = self._distribution(obs)
            ratio = torch.exp(self._log_probs(dist, actions) - old_log_probs)
            return (ratio * advantages).mean()

        def mean_kl() -> torch.Tensor:
            dist = self._distribution(obs)
            return kl_divergence(old_dist, dist).sum(dim=-1).mean() if not self.discrete \
                else kl_divergence(old_dist, dist).mean()

        params = self._actor_params()
        loss = surrogate()
        g = _flat_grad(loss, params, retain_graph=True)

        def fvp(v: torch.Tensor) -> torch.Tensor:
            kl = mean_kl()
            kl_grad = _flat_grad(kl, params, create_graph=True)
            kl_v = (kl_grad * v).sum()
            hv = _flat_grad(kl_v, params, retain_graph=True)
            return hv + self.cg_damping * v

        stepdir = self._conjugate_gradient(fvp, g)
        shs = 0.5 * torch.dot(stepdir, fvp(stepdir))
        lm = torch.sqrt(shs / self.max_kl + 1e-8)
        full_step = stepdir / lm
        expected = torch.dot(g, full_step)

        old_params = self._get_flat_params()
        old_surr = float(loss.item())
        accepted = False
        step_frac = 0.0
        for i in range(self.backtrack_iters):
            frac = self.backtrack_coeff ** i
            self._set_flat_params(old_params + frac * full_step)
            with torch.no_grad():
                new_surr = float(surrogate().item())
                new_kl = float(mean_kl().item())
            if new_kl <= self.max_kl and new_surr > old_surr:
                accepted = True
                step_frac = frac
                break
        if not accepted:
            self._set_flat_params(old_params)

        # value-function regression (a few full-batch gradient steps)
        vf_loss = 0.0
        for _ in range(self.vf_iters):
            values = self.critic(obs)
            v_loss = ((values - batch.returns) ** 2).mean()
            self.vf_opt.zero_grad()
            v_loss.backward()
            self.vf_opt.step()
            vf_loss = float(v_loss.item())

        with torch.no_grad():
            final_kl = float(mean_kl().item())
        return {"surrogate": old_surr, "value_loss": vf_loss, "kl": final_kl,
                "step_frac": step_frac, "expected_improve": float(expected.item())}
