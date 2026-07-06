"""CQL: Conservative Q-Learning for offline RL (Kumar et al., 2020).

Continuous-control CQL(H) built on a SAC backbone: the critic loss adds a
conservative penalty that pushes down Q-values of out-of-distribution actions
(a log-sum-exp over random + policy actions) while pushing up Q-values of the
dataset actions. This prevents value overestimation on actions absent from the
fixed dataset. Trains from a :class:`~reinforce.data.TransitionDataset`.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F

from ..core.agent import BaseAgent
from ..core.env import Env
from ..core.spaces import is_discrete
from ..data import TransitionDataset
from ..networks.policies import SquashedGaussianActor
from ..networks.value import ContinuousQ
from ..utils.torch_utils import get_device, hard_update, soft_update

__all__ = ["CQL"]


class CQL(BaseAgent):
    def __init__(
        self,
        env: Env,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        hidden_sizes: Sequence[int] = (256, 256),
        cql_alpha: float = 1.0,
        n_random: int = 10,
        ent_coef: Union[str, float] = "auto",
        batch_size: int = 256,
        device: str = "auto",
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(env, seed=seed, **kwargs)
        assert not is_discrete(self.action_space), "CQL requires a continuous (Box) action space"
        self.device = get_device(device)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.cql_alpha = float(cql_alpha)
        self.n_random = int(n_random)
        self.batch_size = int(batch_size)
        self.hidden_sizes = tuple(hidden_sizes)

        self.obs_dim = int(np.prod(self.observation_space.shape))
        self.act_dim = int(self.action_space.shape[0])
        low = np.asarray(self.action_space.low, dtype=np.float32)
        high = np.asarray(self.action_space.high, dtype=np.float32)
        self.action_low, self.action_high = low, high

        self.actor = SquashedGaussianActor(self.obs_dim, self.act_dim, low, high, self.hidden_sizes).to(self.device)

        def make_q():
            return ContinuousQ(self.obs_dim, self.act_dim, self.hidden_sizes).to(self.device)

        self.q1, self.q2 = make_q(), make_q()
        self.q1_target, self.q2_target = make_q(), make_q()
        hard_update(self.q1, self.q1_target)
        hard_update(self.q2, self.q2_target)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.q_opt = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=learning_rate)

        self.autotune = ent_coef == "auto"
        if self.autotune:
            self.target_entropy = float(-self.act_dim)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=learning_rate)
            self.alpha = float(self.log_alpha.exp().item())
        else:
            self.alpha = float(ent_coef)
        # uniform density over the [-1, 1]^act_dim tanh support (log 0.5 per dim)
        self._random_log_prob = self.act_dim * math.log(0.5)

    @torch.no_grad()
    def predict(self, obs, deterministic: bool = True) -> np.ndarray:
        obs_t = torch.as_tensor(np.asarray(obs, np.float32).reshape(1, -1), device=self.device)
        action, _, deterministic_action = self.actor.sample(obs_t)
        chosen = deterministic_action if deterministic else action
        return chosen.cpu().numpy()[0].astype(np.float32)

    def _cql_term(self, q_net, obs, action, obs_rep, random_actions, curr_a, curr_lp, next_a, next_lp):
        bsz, n = obs.shape[0], self.n_random
        q_rand = q_net(obs_rep, random_actions).view(bsz, n) - self._random_log_prob
        q_curr = q_net(obs_rep, curr_a).view(bsz, n) - curr_lp.view(bsz, n)
        q_next = q_net(obs_rep, next_a).view(bsz, n) - next_lp.view(bsz, n)
        cat_q = torch.cat([q_rand, q_curr, q_next], dim=1)  # (B, 3N)
        return (torch.logsumexp(cat_q, dim=1) - q_net(obs, action)).mean()

    def _offline_update(self, batch) -> dict:
        obs, action, next_obs = batch.obs, batch.actions, batch.next_obs
        bsz, n = obs.shape[0], self.n_random

        with torch.no_grad():
            next_action, next_logp, _ = self.actor.sample(next_obs)
            next_logp = next_logp.squeeze(-1)
            target_q = torch.min(self.q1_target(next_obs, next_action), self.q2_target(next_obs, next_action))
            y = batch.rewards + batch.discounts * (1.0 - batch.dones) * (target_q - self.alpha * next_logp)

        q1, q2 = self.q1(obs, action), self.q2(obs, action)
        bellman = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        # conservative penalty over random + policy actions
        obs_rep = obs.repeat_interleave(n, dim=0)
        next_rep = next_obs.repeat_interleave(n, dim=0)
        random_actions = torch.empty(bsz * n, self.act_dim, device=self.device).uniform_(-1.0, 1.0)
        curr_a, curr_lp, _ = self.actor.sample(obs_rep)
        next_a, next_lp, _ = self.actor.sample(next_rep)
        cql1 = self._cql_term(self.q1, obs, action, obs_rep, random_actions, curr_a.detach(), curr_lp.detach(), next_a.detach(), next_lp.detach())
        cql2 = self._cql_term(self.q2, obs, action, obs_rep, random_actions, curr_a.detach(), curr_lp.detach(), next_a.detach(), next_lp.detach())
        critic_loss = bellman + self.cql_alpha * (cql1 + cql2)

        self.q_opt.zero_grad()
        critic_loss.backward()
        self.q_opt.step()

        # actor (SAC objective)
        a, logp, _ = self.actor.sample(obs)
        logp = logp.squeeze(-1)
        q = torch.min(self.q1(obs, a), self.q2(obs, a))
        actor_loss = (self.alpha * logp - q).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        metrics = {"critic_loss": float(bellman.item()), "cql": float((cql1 + cql2).item()), "actor_loss": float(actor_loss.item())}
        if self.autotune:
            alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            self.alpha = float(self.log_alpha.exp().item())

        soft_update(self.q1, self.q1_target, self.tau)
        soft_update(self.q2, self.q2_target, self.tau)
        return metrics

    def learn_offline(self, dataset: TransitionDataset, total_steps: int, callback=None, log_interval: int = 1000) -> "CQL":
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

    def learn(self, *args, **kwargs):  # pragma: no cover - guard
        raise NotImplementedError("CQL is offline; use learn_offline(dataset, total_steps).")

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
                "config": dict(gamma=self.gamma, tau=self.tau, hidden_sizes=self.hidden_sizes,
                               cql_alpha=self.cql_alpha, n_random=self.n_random, batch_size=self.batch_size,
                               ent_coef="auto" if self.autotune else self.alpha),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, env: Env = None, device: str = "auto", **kwargs) -> "CQL":
        checkpoint = torch.load(path, map_location=get_device(device), weights_only=False)
        agent = cls(env, device=device, **{**checkpoint["config"], **kwargs})
        agent.actor.load_state_dict(checkpoint["actor"])
        agent.q1.load_state_dict(checkpoint["q1"])
        agent.q2.load_state_dict(checkpoint["q2"])
        hard_update(agent.q1, agent.q1_target)
        hard_update(agent.q2, agent.q2_target)
        return agent
