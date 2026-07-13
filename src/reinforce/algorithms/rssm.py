"""Recurrent State-Space Model (RSSM) world model + Dreamer-style agent.

The RSSM (Hafner et al., PlaNet/Dreamer) is the standard modern world model: a
**deterministic recurrent state** ``h`` (a GRU over the previous stochastic state
and action) plus a **stochastic latent** ``z`` with a learned prior ``p(z|h)`` and
an observation-conditioned posterior ``q(z|h, o)``. It is trained on sequences by
maximizing an ELBO — reconstruction + reward prediction + a KL term between
posterior and prior (with free nats) — and lets the agent *imagine* rollouts in
latent space for actor-critic learning.

.. note:: **Experimental for control.** The RSSM world model demonstrably learns
   the dynamics (reconstruction/reward losses drop, KL is well-behaved); the
   imagination actor-critic is included but not tuned to beat the model-free
   agents. Use :class:`~reinforce.algorithms.MBPO` for a robust model-based agent.
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence

from ..core.agent import BaseAgent
from ..core.env import Env
from ..core.spaces import is_discrete
from ..networks.mlp import build_mlp
from ..networks.policies import SquashedGaussianActor
from ..networks.value import VNetwork
from ..utils.torch_utils import get_device, to_tensor

__all__ = ["RSSM", "DreamerRSSM"]


class RSSM(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, deter: int = 64, stoch: int = 32,
                 hidden: int = 128, free_nats: float = 1.0) -> None:
        super().__init__()
        self.deter, self.stoch, self.free_nats = deter, stoch, free_nats
        self.embed = build_mlp(obs_dim, hidden, (hidden,), activation=nn.ELU)
        self.gru = nn.GRUCell(stoch + act_dim, deter)
        self.prior_net = build_mlp(deter, 2 * stoch, (hidden,), activation=nn.ELU)
        self.post_net = build_mlp(deter + hidden, 2 * stoch, (hidden,), activation=nn.ELU)
        self.decoder = build_mlp(deter + stoch, obs_dim, (hidden,), activation=nn.ELU)
        self.reward = build_mlp(deter + stoch, 1, (hidden,), activation=nn.ELU)

    def _dist(self, params: torch.Tensor) -> Normal:
        mean, std = torch.chunk(params, 2, dim=-1)
        return Normal(mean, torch.nn.functional.softplus(std) + 0.1)

    def initial(self, batch: int, device):
        return (torch.zeros(batch, self.deter, device=device),
                torch.zeros(batch, self.stoch, device=device))

    def obs_step(self, h, z, a_prev, embed):
        h = self.gru(torch.cat([z, a_prev], dim=-1), h)
        prior = self._dist(self.prior_net(h))
        post = self._dist(self.post_net(torch.cat([h, embed], dim=-1)))
        return h, post.rsample(), prior, post

    def img_step(self, h, z, a_prev):
        h = self.gru(torch.cat([z, a_prev], dim=-1), h)
        prior = self._dist(self.prior_net(h))
        return h, prior.rsample()

    def loss(self, obs_seq: torch.Tensor, act_seq: torch.Tensor, rew_seq: torch.Tensor, kl_scale=1.0):
        b, length = obs_seq.shape[0], obs_seq.shape[1]
        embed = self.embed(obs_seq)
        a_prev = torch.cat([torch.zeros_like(act_seq[:, :1]), act_seq[:, :-1]], dim=1)
        h, z = self.initial(b, obs_seq.device)
        feats, kls = [], []
        for t in range(length):
            h, z, prior, post = self.obs_step(h, z, a_prev[:, t], embed[:, t])
            feats.append(torch.cat([h, z], dim=-1))
            kls.append(kl_divergence(post, prior).sum(dim=-1))
        feat = torch.stack(feats, dim=1)
        recon = ((self.decoder(feat) - obs_seq) ** 2).sum(dim=-1).mean()
        reward = ((self.reward(feat).squeeze(-1) - rew_seq) ** 2).mean()
        kl = torch.stack(kls, dim=1).mean().clamp(min=self.free_nats)
        return recon + reward + kl_scale * kl, {
            "recon": float(recon.item()), "reward": float(reward.item()), "kl": float(kl.item())
        }


class DreamerRSSM(BaseAgent):
    def __init__(
        self,
        env: Env,
        learning_rate: float = 6e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        deter_dim: int = 64,
        stoch_dim: int = 32,
        hidden: int = 128,
        seq_len: int = 20,
        horizon: int = 10,
        batch_size: int = 32,
        ent_coef: float = 1e-3,
        learning_starts: int = 1_000,
        train_freq: int = 100,
        model_updates: int = 30,
        max_episodes: int = 200,
        device: str = "auto",
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(env, seed=seed, **kwargs)
        assert not is_discrete(self.action_space), "DreamerRSSM requires a continuous action space"
        self.device = get_device(device)
        self.gamma, self.gae_lambda = float(gamma), float(gae_lambda)
        self.seq_len, self.horizon = int(seq_len), int(horizon)
        self.batch_size, self.ent_coef = int(batch_size), float(ent_coef)
        self.learning_starts, self.train_freq = int(learning_starts), int(train_freq)
        self.model_updates, self.max_episodes = int(model_updates), int(max_episodes)

        self.obs_dim = int(np.prod(self.observation_space.shape))
        self.act_dim = int(self.action_space.shape[0])
        low = np.asarray(self.action_space.low, dtype=np.float32)
        high = np.asarray(self.action_space.high, dtype=np.float32)
        self.action_low, self.action_high = low, high
        feat_dim = deter_dim + stoch_dim

        self.rssm = RSSM(self.obs_dim, self.act_dim, deter_dim, stoch_dim, hidden).to(self.device)
        self.actor = SquashedGaussianActor(feat_dim, self.act_dim, low, high, (hidden, hidden)).to(self.device)
        self.critic = VNetwork(feat_dim, (hidden, hidden), nn.ELU).to(self.device)
        self.model_opt = torch.optim.Adam(self.rssm.parameters(), lr=learning_rate)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)

        self._episodes: List[dict] = []
        self._cur: dict = {"obs": [], "act": [], "rew": []}
        self.model_losses: List[float] = []
        self._h = self._z = None  # running state for predict()

    # -- sequence storage / sampling --------------------------------------
    def _end_episode(self):
        if len(self._cur["obs"]) > self.seq_len:
            self._episodes.append({k: np.asarray(v, dtype=np.float32) for k, v in self._cur.items()})
            self._episodes = self._episodes[-self.max_episodes:]
        self._cur = {"obs": [], "act": [], "rew": []}

    def _sample_sequences(self):
        eligible = [e for e in self._episodes if len(e["obs"]) > self.seq_len]
        obs, act, rew = [], [], []
        for _ in range(self.batch_size):
            e = eligible[self.rng.integers(len(eligible))]
            s = int(self.rng.integers(0, len(e["obs"]) - self.seq_len))
            sl = slice(s, s + self.seq_len)
            obs.append(e["obs"][sl])
            act.append(e["act"][sl])
            rew.append(e["rew"][sl])
        t = lambda a: torch.as_tensor(np.stack(a), device=self.device)  # noqa: E731
        return t(obs), t(act), t(rew)

    def _train_model(self) -> float:
        loss_val = 0.0
        for _ in range(self.model_updates):
            obs, act, rew = self._sample_sequences()
            loss, info = self.rssm.loss(obs, act, rew)
            self.model_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.rssm.parameters(), 100.0)
            self.model_opt.step()
            loss_val = info["recon"]
        self.model_losses.append(loss_val)
        return loss_val

    def _train_behavior(self):
        obs, act, rew = self._sample_sequences()
        with torch.no_grad():
            embed = self.rssm.embed(obs)
            a_prev = torch.cat([torch.zeros_like(act[:, :1]), act[:, :-1]], dim=1)
            h, z = self.rssm.initial(obs.shape[0], self.device)
            for t in range(self.seq_len):
                h, z, _, _ = self.rssm.obs_step(h, z, a_prev[:, t], embed[:, t])
        h, z = h.detach(), z.detach()

        feats, rewards, entropies = [], [], []
        for _ in range(self.horizon):
            feat = torch.cat([h, z], dim=-1)
            action, logp, _ = self.actor.sample(feat)
            feats.append(feat)
            entropies.append(-logp.squeeze(-1))
            h, z = self.rssm.img_step(h, z, action)
            rewards.append(self.rssm.reward(torch.cat([h, z], dim=-1)).squeeze(-1))
        feats.append(torch.cat([h, z], dim=-1))
        values = [self.critic(f) for f in feats]

        returns_rev = []
        last = values[-1]
        for t in reversed(range(self.horizon)):
            last = rewards[t] + self.gamma * ((1 - self.gae_lambda) * values[t + 1] + self.gae_lambda * last)
            returns_rev.append(last)
        returns_t = torch.stack(returns_rev[::-1])
        actor_loss = -returns_t.mean() - self.ent_coef * torch.stack(entropies).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        value_pred = torch.stack([self.critic(f.detach()) for f in feats[:-1]])
        value_loss = ((value_pred - returns_t.detach()) ** 2).mean()
        self.critic_opt.zero_grad()
        value_loss.backward()
        self.critic_opt.step()
        return {"actor_loss": float(actor_loss.item()), "value_loss": float(value_loss.item())}

    # -- acting ------------------------------------------------------------
    def reset_states(self) -> None:
        self._h = self._z = None

    @torch.no_grad()
    def predict(self, obs, deterministic: bool = True) -> np.ndarray:
        obs_t = to_tensor(np.asarray(obs, np.float32).reshape(1, -1), self.device)
        if self._h is None:
            h, z = self.rssm.initial(1, self.device)
            a_prev = torch.zeros(1, self.act_dim, device=self.device)
        else:
            h, z, a_prev = self._h, self._z, self._last_action
        embed = self.rssm.embed(obs_t)
        h, z, _, _ = self.rssm.obs_step(h, z, a_prev, embed)
        self._h, self._z = h, z
        action, _, det = self.actor.sample(torch.cat([h, z], dim=-1))
        chosen = det if deterministic else action
        self._last_action = chosen
        return chosen.cpu().numpy()[0].astype(np.float32)

    def learn(self, total_steps: int, callback=None, log_interval: int = 10) -> "DreamerRSSM":
        if callback is not None:
            callback.on_training_start(self)
        obs, _ = self.env.reset(seed=self.seed)
        self.reset_states()
        ep_return, episodes = 0.0, 0
        returns_window: deque = deque(maxlen=100)
        metrics: dict = {}
        for step in range(total_steps):
            if step < self.learning_starts or not self._episodes:
                action = self.action_space.sample()
            else:
                action = self.predict(obs, deterministic=False)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            self._cur["obs"].append(np.asarray(obs, np.float32))
            self._cur["act"].append(np.asarray(action, np.float32))
            self._cur["rew"].append(float(reward))
            obs = next_obs
            ep_return += reward
            self.num_timesteps += 1

            if step >= self.learning_starts and step % self.train_freq == 0 and \
                    len([e for e in self._episodes if len(e["obs"]) > self.seq_len]) > 0:
                metrics["recon"] = self._train_model()
                metrics.update(self._train_behavior())

            if callback is not None and not callback.on_step():
                break
            if terminated or truncated:
                self._end_episode()
                episodes += 1
                returns_window.append(ep_return)
                if episodes % log_interval == 0 and returns_window:
                    self.logger.record("rollout/ep_return_mean", float(np.mean(returns_window)))
                    for k, v in metrics.items():
                        self.logger.record(f"train/{k}", v)
                    self.logger.dump(self.num_timesteps)
                obs, _ = self.env.reset()
                self.reset_states()
                ep_return = 0.0
        if callback is not None:
            callback.on_training_end()
        return self

    def save(self, path: str) -> None:
        torch.save({"rssm": self.rssm.state_dict(), "actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict(),
                    "config": dict(gamma=self.gamma, gae_lambda=self.gae_lambda, seq_len=self.seq_len,
                                   horizon=self.horizon, batch_size=self.batch_size)}, path)

    @classmethod
    def load(cls, path: str, env: Env = None, device: str = "auto", **kwargs) -> "DreamerRSSM":
        ckpt = torch.load(path, map_location=get_device(device), weights_only=False)
        agent = cls(env, device=device, **{**ckpt["config"], **kwargs})
        agent.rssm.load_state_dict(ckpt["rssm"])
        agent.actor.load_state_dict(ckpt["actor"])
        agent.critic.load_state_dict(ckpt["critic"])
        return agent
