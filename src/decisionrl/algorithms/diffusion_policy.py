"""Diffusion Policy (Chi et al., 2023): actions from a conditional denoiser.

Represents the policy as a **conditional denoising diffusion model** over the
continuous action: a network learns to denoise noisy actions conditioned on the
observation, and actions are produced by running the reverse diffusion chain from
Gaussian noise. Trained by behavior cloning on a demonstration dataset, it can
capture multimodal action distributions that a unimodal Gaussian policy cannot —
the current default for continuous-control imitation in robotics.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.agent import BaseAgent
from ..core.env import Env
from ..data import TransitionDataset
from ..networks.mlp import build_mlp
from ..utils.torch_utils import get_device, to_tensor

__all__ = ["DiffusionPolicy"]


def _timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class _EpsNet(nn.Module):
    """Predicts the noise added to an action, conditioned on obs and timestep."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Sequence[int], time_dim: int = 32):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.net = build_mlp(act_dim + obs_dim + time_dim, act_dim, hidden_sizes)

    def forward(self, action: torch.Tensor, obs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.time_mlp(_timestep_embedding(t, self.time_dim))
        return self.net(torch.cat([action, obs, temb], dim=-1))


class DiffusionPolicy(BaseAgent):
    def __init__(
        self,
        env: Env,
        n_diffusion_steps: int = 20,
        hidden_sizes: Sequence[int] = (128, 128),
        learning_rate: float = 1e-3,
        device: str = "auto",
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(env, seed=seed, **kwargs)
        self.device = get_device(device)
        self.T = int(n_diffusion_steps)
        self.obs_dim = int(np.prod(self.observation_space.shape))
        self.act_dim = int(self.action_space.shape[0])
        self.action_low = np.asarray(self.action_space.low, dtype=np.float32)
        self.action_high = np.asarray(self.action_space.high, dtype=np.float32)
        self.hidden_sizes = tuple(hidden_sizes)

        # linear DDPM noise schedule
        betas = torch.linspace(1e-4, 0.02, self.T, device=self.device)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        self.model = _EpsNet(self.obs_dim, self.act_dim, self.hidden_sizes).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self, dataset: TransitionDataset, n_iters: int = 3000, batch_size: int = 128,
              log_interval: int = 0) -> dict:
        """Behavior-clone the dataset's actions with a denoising diffusion loss."""
        from collections import deque

        losses: deque = deque(maxlen=100)
        for it in range(n_iters):
            batch = dataset.sample(batch_size)
            obs, a0 = batch.obs, batch.actions.reshape(batch_size, self.act_dim)
            t = torch.randint(0, self.T, (batch_size,), device=self.device)
            noise = torch.randn_like(a0)
            abar = self.alpha_bars[t].unsqueeze(1)
            a_t = torch.sqrt(abar) * a0 + torch.sqrt(1 - abar) * noise
            pred = self.model(a_t, obs, t)
            loss = F.mse_loss(pred, noise)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.item()))
            if log_interval and (it + 1) % log_interval == 0:
                self.logger.record("diffusion/loss", float(np.mean(losses)))
                self.logger.dump(it + 1)
        return {"loss": float(np.mean(losses))}

    @torch.no_grad()
    def _sample(self, obs_t: torch.Tensor, deterministic: bool) -> torch.Tensor:
        # In deterministic mode, seed the initial noise and drop the per-step noise
        # so the same observation maps to a reproducible action (needed for stable
        # evaluation and save/load parity); diffusion sampling is otherwise random.
        gen = None
        if deterministic:
            gen = torch.Generator(device=self.device)
            gen.manual_seed(0)
        a = torch.randn(obs_t.shape[0], self.act_dim, device=self.device, generator=gen)
        for t in reversed(range(self.T)):
            t_batch = torch.full((obs_t.shape[0],), t, device=self.device, dtype=torch.long)
            eps = self.model(a, obs_t, t_batch)
            alpha, abar, beta = self.alphas[t], self.alpha_bars[t], self.betas[t]
            mean = (a - beta / torch.sqrt(1 - abar) * eps) / torch.sqrt(alpha)
            if t > 0 and not deterministic:
                a = mean + torch.sqrt(beta) * torch.randn_like(a)
            else:
                a = mean
        return a

    def predict(self, obs, deterministic: bool = True):
        obs_t = to_tensor(np.asarray(obs, dtype=np.float32).reshape(1, -1), self.device)
        action = self._sample(obs_t, deterministic).cpu().numpy()[0]
        return np.clip(action, self.action_low, self.action_high)

    def learn(self, *args, **kwargs):  # pragma: no cover - guard
        raise NotImplementedError("DiffusionPolicy is imitation-based; use train(dataset).")

    def save(self, path: str) -> None:
        torch.save({"model": self.model.state_dict(),
                    "config": dict(n_diffusion_steps=self.T, hidden_sizes=self.hidden_sizes)}, path)

    @classmethod
    def load(cls, path: str, env: Env = None, device: str = "auto", **kwargs) -> "DiffusionPolicy":
        checkpoint = torch.load(path, map_location=get_device(device), weights_only=False)
        agent = cls(env, device=device, **{**checkpoint["config"], **kwargs})
        agent.model.load_state_dict(checkpoint["model"])
        return agent
