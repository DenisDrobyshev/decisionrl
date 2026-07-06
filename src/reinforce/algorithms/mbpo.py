"""MBPO: Model-Based Policy Optimization (Janner et al., 2019).

Learns an ensemble dynamics model from real data, generates short synthetic
rollouts branched off real states, and trains SAC on a mix of real and model
transitions - buying large sample-efficiency gains over model-free SAC. Built on
top of the SAC agent (reused actor/critic/temperature update).
"""

from __future__ import annotations

from collections import deque
from typing import Optional, Sequence

import numpy as np
import torch

from ..buffers.replay import ReplayBatch, ReplayBuffer
from ..core.env import Env
from ..networks.dynamics import EnsembleDynamics
from .sac import SAC

__all__ = ["MBPO"]


class MBPO(SAC):
    def __init__(
        self,
        env: Env,
        ensemble_size: int = 5,
        model_hidden_sizes: Sequence[int] = (200, 200),
        model_lr: float = 1e-3,
        model_train_freq: int = 250,
        model_train_batches: int = 64,
        rollout_length: int = 1,
        model_rollouts: int = 400,
        real_ratio: float = 0.1,
        model_buffer_size: int = 100_000,
        device: str = "auto",
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(env, device=device, seed=seed, **kwargs)
        self.model_train_freq = int(model_train_freq)
        self.model_train_batches = int(model_train_batches)
        self.rollout_length = int(rollout_length)
        self.model_rollouts = int(model_rollouts)
        self.real_ratio = float(real_ratio)

        self.dynamics = EnsembleDynamics(self.obs_dim, self.act_dim, ensemble_size, model_hidden_sizes).to(self.device)
        self.model_opt = torch.optim.Adam(self.dynamics.parameters(), lr=model_lr)
        self.model_buffer = ReplayBuffer(
            model_buffer_size, self.observation_space, self.action_space,
            device=str(self.device), seed=seed,
        )
        self._obs_low = torch.as_tensor(self.observation_space.low, device=self.device, dtype=torch.float32)
        self._obs_high = torch.as_tensor(self.observation_space.high, device=self.device, dtype=torch.float32)

    # -- mixed real + model batch for the SAC update ----------------------
    def _sample(self):
        if len(self.model_buffer) == 0:
            return self.buffer.sample(self.batch_size)
        n_model = int(self.batch_size * (1.0 - self.real_ratio))
        n_real = self.batch_size - n_model
        real = self.buffer.sample(max(1, n_real))
        model = self.model_buffer.sample(max(1, n_model))

        def cat(a, b):
            return torch.cat([a, b], dim=0)

        return ReplayBatch(
            obs=cat(real.obs, model.obs), actions=cat(real.actions, model.actions),
            rewards=cat(real.rewards, model.rewards), next_obs=cat(real.next_obs, model.next_obs),
            dones=cat(real.dones, model.dones), discounts=cat(real.discounts, model.discounts),
        )

    def _train_ensemble(self) -> float:
        if len(self.buffer) < 256:
            return 0.0
        loss_val = 0.0
        for _ in range(self.model_train_batches):
            b = self.buffer.sample(256)
            target = torch.cat([b.next_obs - b.obs, b.rewards.unsqueeze(-1)], dim=-1)
            preds = self.dynamics.forward_all(b.obs, b.actions)  # (E, B, obs_dim+1)
            loss = ((preds - target.unsqueeze(0)) ** 2).mean()
            self.model_opt.zero_grad()
            loss.backward()
            self.model_opt.step()
            loss_val = float(loss.item())
        return loss_val

    @torch.no_grad()
    def _generate_rollouts(self) -> None:
        n = min(self.model_rollouts, len(self.buffer))
        if n == 0:
            return
        state = self.buffer.sample(n).obs
        for _ in range(self.rollout_length):
            action, _, _ = self.actor.sample(state)
            member = int(self.rng.integers(self.dynamics.ensemble_size))
            next_state, reward = self.dynamics.predict(state, action, member)
            next_state = torch.clamp(next_state, self._obs_low, self._obs_high)
            s, a = state.cpu().numpy(), action.cpu().numpy()
            r, ns = reward.cpu().numpy(), next_state.cpu().numpy()
            for i in range(n):
                self.model_buffer.add(s[i], a[i], float(r[i]), ns[i], False)
            state = next_state

    def learn(self, total_steps: int, callback=None, log_interval: int = 10) -> "MBPO":
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
                action = self.act(obs, deterministic=False)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            self.buffer.add(obs, action, reward, next_obs, terminated,
                            episode_end=(terminated or truncated))
            obs = next_obs
            ep_return += reward
            self.num_timesteps += 1

            if step >= self.learning_starts:
                if step % self.model_train_freq == 0:
                    metrics["model_loss"] = self._train_ensemble()
                    self._generate_rollouts()
                for _ in range(self.gradient_steps):
                    metrics.update(self.train_step())

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
