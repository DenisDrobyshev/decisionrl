"""IQL: Implicit Q-Learning for offline RL (Kostrikov et al., 2022).

Avoids querying out-of-distribution actions entirely: a value function is fit to
an *expectile* of Q over the dataset, Q is regressed toward ``r + gamma V(s')``,
and the policy is extracted by advantage-weighted regression. Trains from a fixed
:class:`~reinforce.data.TransitionDataset` with no environment interaction.
"""

from __future__ import annotations

from collections import deque
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.agent import BaseAgent
from ..core.env import Env
from ..core.spaces import is_discrete
from ..data import TransitionDataset
from ..networks.policies import GaussianActor
from ..networks.value import ContinuousQ, VNetwork
from ..utils.torch_utils import get_device, hard_update, soft_update, to_tensor

__all__ = ["IQL"]


class IQL(BaseAgent):
    def __init__(
        self,
        env: Env,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        hidden_sizes: Sequence[int] = (256, 256),
        expectile: float = 0.7,
        beta: float = 3.0,
        max_weight: float = 100.0,
        batch_size: int = 256,
        device: str = "auto",
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(env, seed=seed, **kwargs)
        assert not is_discrete(self.action_space), "IQL requires a continuous (Box) action space"
        self.device = get_device(device)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.expectile = float(expectile)
        self.beta = float(beta)
        self.max_weight = float(max_weight)
        self.batch_size = int(batch_size)
        self.hidden_sizes = tuple(hidden_sizes)

        self.obs_dim = int(np.prod(self.observation_space.shape))
        self.act_dim = int(self.action_space.shape[0])
        self.action_low = np.asarray(self.action_space.low, dtype=np.float32)
        self.action_high = np.asarray(self.action_space.high, dtype=np.float32)

        self.actor = GaussianActor(self.obs_dim, self.act_dim, self.hidden_sizes, nn.ReLU).to(self.device)

        def make_q():
            return ContinuousQ(self.obs_dim, self.act_dim, self.hidden_sizes).to(self.device)

        self.q1, self.q2 = make_q(), make_q()
        self.q1_target, self.q2_target = make_q(), make_q()
        hard_update(self.q1, self.q1_target)
        hard_update(self.q2, self.q2_target)
        self.value = VNetwork(self.obs_dim, self.hidden_sizes, nn.ReLU).to(self.device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.q_opt = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=learning_rate)
        self.v_opt = torch.optim.Adam(self.value.parameters(), lr=learning_rate)

    @torch.no_grad()
    def predict(self, obs, deterministic: bool = True) -> np.ndarray:
        obs_t = to_tensor(np.asarray(obs, dtype=np.float32).reshape(1, -1), self.device)
        dist = self.actor(obs_t)
        action = dist.mean if deterministic else dist.sample()
        return np.clip(action.cpu().numpy()[0], self.action_low, self.action_high)

    def _expectile_loss(self, diff: torch.Tensor) -> torch.Tensor:
        weight = torch.where(diff > 0, self.expectile, 1.0 - self.expectile)
        return (weight * diff.pow(2)).mean()

    def _offline_update(self, batch) -> dict:
        # Value: fit an expectile of the (target) Q over dataset actions.
        with torch.no_grad():
            q_target = torch.min(self.q1_target(batch.obs, batch.actions), self.q2_target(batch.obs, batch.actions))
        value = self.value(batch.obs)
        v_loss = self._expectile_loss(q_target - value)
        self.v_opt.zero_grad()
        v_loss.backward()
        self.v_opt.step()

        # Q: regress toward r + gamma * V(s').
        with torch.no_grad():
            next_v = self.value(batch.next_obs)
            q_backup = batch.rewards + batch.discounts * (1.0 - batch.dones) * next_v
        q1 = self.q1(batch.obs, batch.actions)
        q2 = self.q2(batch.obs, batch.actions)
        q_loss = F.mse_loss(q1, q_backup) + F.mse_loss(q2, q_backup)
        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        # Policy: advantage-weighted regression.
        with torch.no_grad():
            adv = q_target - self.value(batch.obs)
            weight = torch.exp(self.beta * adv).clamp(max=self.max_weight)
        log_prob = self.actor(batch.obs).log_prob(batch.actions).sum(dim=-1)
        actor_loss = -(weight * log_prob).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        soft_update(self.q1, self.q1_target, self.tau)
        soft_update(self.q2, self.q2_target, self.tau)
        return {
            "v_loss": float(v_loss.item()),
            "q_loss": float(q_loss.item()),
            "actor_loss": float(actor_loss.item()),
        }

    def learn_offline(
        self, dataset: TransitionDataset, total_steps: int, callback=None, log_interval: int = 1000
    ) -> "IQL":
        if callback is not None:
            callback.on_training_start(self)
        losses: deque = deque(maxlen=100)
        for _ in range(total_steps):
            batch = dataset.sample(self.batch_size)
            metrics = self._offline_update(batch)
            losses.append(metrics["q_loss"])
            self.num_timesteps += 1
            if callback is not None and not callback.on_step():
                break
            if self.num_timesteps % log_interval == 0:
                self.logger.record("train/q_loss", float(np.mean(losses)))
                self.logger.dump(self.num_timesteps)
        if callback is not None:
            callback.on_training_end()
        return self

    def learn(self, *args, **kwargs):  # pragma: no cover - guard
        raise NotImplementedError("IQL is offline; use learn_offline(dataset, total_steps).")

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
                "value": self.value.state_dict(),
                "config": dict(gamma=self.gamma, tau=self.tau, hidden_sizes=self.hidden_sizes,
                               expectile=self.expectile, beta=self.beta, batch_size=self.batch_size),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, env: Env = None, device: str = "auto", **kwargs) -> "IQL":
        checkpoint = torch.load(path, map_location=get_device(device), weights_only=False)
        agent = cls(env, device=device, **{**checkpoint["config"], **kwargs})
        agent.actor.load_state_dict(checkpoint["actor"])
        agent.q1.load_state_dict(checkpoint["q1"])
        agent.q2.load_state_dict(checkpoint["q2"])
        agent.value.load_state_dict(checkpoint["value"])
        hard_update(agent.q1, agent.q1_target)
        hard_update(agent.q2, agent.q2_target)
        return agent
