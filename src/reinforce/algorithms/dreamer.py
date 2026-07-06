"""Dreamer-style latent world model with actor-critic learned in imagination.

.. note::
   **Experimental.** The latent world model learns the dynamics well, but the
   pure imagination-gradient policy learning (deterministic latents, no RSSM/KL)
   is not tuned to be competitive with the model-free agents on these tasks - it
   is included to demonstrate the world-model + imagination machinery. For a
   robust model-based agent use :class:`~reinforce.algorithms.MBPO`.

A compact Dreamer (Hafner et al., 2020) for low-dimensional continuous control:
a learned latent world model (encoder / latent transition / reward / decoder) is
trained from replay, then the actor and critic are trained *entirely in
imagination* by rolling the latent dynamics forward and back-propagating
analytic (pathwise) gradients of the lambda-returns through the differentiable
model. This is a simplified deterministic-latent variant (no full RSSM/KL), aimed
at demonstrating model-based imagination on vector observations.
"""

from __future__ import annotations

from collections import deque
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from ..buffers.replay import ReplayBuffer
from ..core.agent import BaseAgent
from ..core.env import Env
from ..core.spaces import is_discrete
from ..networks.mlp import build_mlp
from ..networks.policies import SquashedGaussianActor
from ..networks.value import VNetwork
from ..utils.torch_utils import get_device, to_tensor

__all__ = ["Dreamer"]


class _WorldModel(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, latent_dim: int, hidden: Sequence[int]) -> None:
        super().__init__()
        self.encoder = build_mlp(obs_dim, latent_dim, hidden, activation=nn.ELU)
        self.transition = build_mlp(latent_dim + act_dim, latent_dim, hidden, activation=nn.ELU)
        self.reward = build_mlp(latent_dim + act_dim, 1, hidden, activation=nn.ELU)
        self.decoder = build_mlp(latent_dim, obs_dim, hidden, activation=nn.ELU)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def step(self, latent: torch.Tensor, action: torch.Tensor):
        x = torch.cat([latent, action], dim=-1)
        return self.transition(x), self.reward(x).squeeze(-1)


class Dreamer(BaseAgent):
    def __init__(
        self,
        env: Env,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        latent_dim: int = 32,
        hidden_sizes: Sequence[int] = (200, 200),
        horizon: int = 5,
        ent_coef: float = 1e-3,
        buffer_size: int = 100_000,
        batch_size: int = 256,
        learning_starts: int = 1_000,
        train_freq: int = 50,
        model_updates: int = 20,
        device: str = "auto",
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(env, seed=seed, **kwargs)
        assert not is_discrete(self.action_space), "Dreamer requires a continuous (Box) action space"
        self.device = get_device(device)
        self.gamma, self.gae_lambda = float(gamma), float(gae_lambda)
        self.horizon = int(horizon)
        self.ent_coef = float(ent_coef)
        self.batch_size = int(batch_size)
        self.learning_starts = int(learning_starts)
        self.train_freq = int(train_freq)
        self.model_updates = int(model_updates)

        self.obs_dim = int(np.prod(self.observation_space.shape))
        self.act_dim = int(self.action_space.shape[0])
        low = np.asarray(self.action_space.low, dtype=np.float32)
        high = np.asarray(self.action_space.high, dtype=np.float32)
        self.action_low, self.action_high = low, high

        self.world = _WorldModel(self.obs_dim, self.act_dim, latent_dim, hidden_sizes).to(self.device)
        # the actor/critic operate on latent states produced by the world model
        self.actor_latent = SquashedGaussianActor(latent_dim, self.act_dim, low, high, hidden_sizes).to(self.device)
        self.critic = VNetwork(latent_dim, hidden_sizes, nn.ELU).to(self.device)

        self.world_opt = torch.optim.Adam(self.world.parameters(), lr=learning_rate)
        self.actor_opt = torch.optim.Adam(self.actor_latent.parameters(), lr=learning_rate)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.buffer = ReplayBuffer(buffer_size, self.observation_space, self.action_space,
                                   device=str(self.device), seed=seed, gamma=gamma)

    @torch.no_grad()
    def predict(self, obs, deterministic: bool = True) -> np.ndarray:
        obs_t = to_tensor(np.asarray(obs, np.float32).reshape(1, -1), self.device)
        latent = self.world.encode(obs_t)
        action, _, det = self.actor_latent.sample(latent)
        chosen = det if deterministic else action
        return chosen.cpu().numpy()[0].astype(np.float32)

    def _train_world(self) -> float:
        loss_val = 0.0
        for _ in range(self.model_updates):
            b = self.buffer.sample(self.batch_size)
            s = self.world.encode(b.obs)
            s_next = self.world.encode(b.next_obs)
            s_pred, r_pred = self.world.step(s, b.actions)
            loss = (
                ((self.world.decoder(s) - b.obs) ** 2).mean()
                + ((self.world.decoder(s_next) - b.next_obs) ** 2).mean()
                + ((s_pred - s_next.detach()) ** 2).mean()
                + ((r_pred - b.rewards) ** 2).mean()
            )
            self.world_opt.zero_grad()
            loss.backward()
            self.world_opt.step()
            loss_val = float(loss.item())
        return loss_val

    def _train_behavior(self) -> dict:
        b = self.buffer.sample(self.batch_size)
        with torch.no_grad():
            latent = self.world.encode(b.obs)

        states, rewards, entropies = [], [], []
        s = latent
        for _ in range(self.horizon):
            action, logp, _ = self.actor_latent.sample(s)  # reparameterized (pathwise grad)
            states.append(s)
            entropies.append(-logp.squeeze(-1))
            s, r = self.world.step(s, action)
            rewards.append(r)
        states.append(s)
        values = [self.critic(st) for st in states]

        # lambda-returns over the imagined trajectory
        returns = [None] * self.horizon
        last = values[-1]
        for t in reversed(range(self.horizon)):
            last = rewards[t] + self.gamma * ((1 - self.gae_lambda) * values[t + 1] + self.gae_lambda * last)
            returns[t] = last
        returns_t = torch.stack(returns)  # (H, B)

        entropy = torch.stack(entropies).mean()
        actor_loss = -returns_t.mean() - self.ent_coef * entropy
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        value_pred = torch.stack([self.critic(st.detach()) for st in states[:-1]])
        value_loss = ((value_pred - returns_t.detach()) ** 2).mean()
        self.critic_opt.zero_grad()
        value_loss.backward()
        self.critic_opt.step()
        return {"actor_loss": float(actor_loss.item()), "value_loss": float(value_loss.item())}

    def learn(self, total_steps: int, callback=None, log_interval: int = 10) -> "Dreamer":
        self._total_timesteps = self.num_timesteps + total_steps
        if callback is not None:
            callback.on_training_start(self)
        obs, _ = self.env.reset(seed=self.seed)
        ep_return, episodes = 0.0, 0
        returns_window: deque = deque(maxlen=100)
        metrics: dict = {}

        for step in range(total_steps):
            if step < self.learning_starts:
                action = self.action_space.sample()
            else:
                action = self.predict(obs, deterministic=False)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            self.buffer.add(obs, action, reward, next_obs, terminated,
                            episode_end=(terminated or truncated))
            obs = next_obs
            ep_return += reward
            self.num_timesteps += 1

            if step >= self.learning_starts and step % self.train_freq == 0:
                metrics["model_loss"] = self._train_world()
                metrics.update(self._train_behavior())

            if callback is not None and not callback.on_step():
                break

            if terminated or truncated:
                episodes += 1
                returns_window.append(ep_return)
                if episodes % log_interval == 0:
                    self.logger.record("rollout/ep_return_mean", float(np.mean(returns_window)))
                    for k, v in metrics.items():
                        self.logger.record(f"train/{k}", v)
                    self.logger.dump(self.num_timesteps)
                obs, _ = self.env.reset()
                ep_return = 0.0

        if callback is not None:
            callback.on_training_end()
        return self

    def save(self, path: str) -> None:
        torch.save({"world": self.world.state_dict(), "actor": self.actor_latent.state_dict(),
                    "critic": self.critic.state_dict(),
                    "config": dict(gamma=self.gamma, gae_lambda=self.gae_lambda, horizon=self.horizon,
                                   batch_size=self.batch_size, learning_starts=self.learning_starts)}, path)

    @classmethod
    def load(cls, path: str, env: Env = None, device: str = "auto", **kwargs) -> "Dreamer":
        ckpt = torch.load(path, map_location=get_device(device), weights_only=False)
        agent = cls(env, device=device, **{**ckpt["config"], **kwargs})
        agent.world.load_state_dict(ckpt["world"])
        agent.actor_latent.load_state_dict(ckpt["actor"])
        agent.critic.load_state_dict(ckpt["critic"])
        return agent
