"""REINFORCE: Monte-Carlo policy gradient (Williams, 1992).

The foundational policy-gradient method. Supports discrete and continuous
actions and an optional learned value baseline to reduce variance. Updates once
per completed episode from full-episode returns.
"""

from __future__ import annotations

from collections import deque
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from ..core.agent import BaseAgent
from ..core.env import Env
from ..core.spaces import is_discrete
from ..networks.policies import CategoricalActor, GaussianActor
from ..networks.value import VNetwork
from ..utils.torch_utils import get_device, to_tensor

__all__ = ["REINFORCE"]


class REINFORCE(BaseAgent):
    def __init__(
        self,
        env: Env,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: str = "tanh",
        baseline: bool = True,
        ent_coef: float = 0.0,
        normalize_returns: bool = True,
        device: str = "auto",
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(env, seed=seed, **kwargs)
        self.device = get_device(device)
        self.gamma = float(gamma)
        self.baseline = bool(baseline)
        self.ent_coef = float(ent_coef)
        self.normalize_returns = bool(normalize_returns)
        self.hidden_sizes = tuple(hidden_sizes)

        act_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation]
        self.discrete = is_discrete(self.action_space)
        self.obs_dim = int(np.prod(self.observation_space.shape))
        if self.discrete:
            self.n_actions = int(self.action_space.n)
            self.actor = CategoricalActor(self.obs_dim, self.n_actions, self.hidden_sizes, act_fn)
        else:
            self.act_dim = int(self.action_space.shape[0])
            self.action_low = np.asarray(self.action_space.low, dtype=np.float32)
            self.action_high = np.asarray(self.action_space.high, dtype=np.float32)
            self.actor = GaussianActor(self.obs_dim, self.act_dim, self.hidden_sizes, act_fn)
        self.actor.to(self.device)

        params = list(self.actor.parameters())
        if self.baseline:
            self.critic = VNetwork(self.obs_dim, self.hidden_sizes, act_fn).to(self.device)
            params += list(self.critic.parameters())
        else:
            self.critic = None
        self.optimizer = torch.optim.Adam(params, lr=learning_rate)

    @torch.no_grad()
    def predict(self, obs, deterministic: bool = True):
        obs_t = to_tensor(np.asarray(obs, dtype=np.float32).reshape(1, -1), self.device)
        dist = self.actor(obs_t)
        if self.discrete:
            action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
            return int(action.item())
        action = dist.mean if deterministic else dist.sample()
        return np.clip(action.cpu().numpy()[0], self.action_low, self.action_high)

    def _discounted_returns(self, rewards) -> np.ndarray:
        returns = np.zeros(len(rewards), dtype=np.float32)
        running = 0.0
        for t in reversed(range(len(rewards))):
            running = rewards[t] + self.gamma * running
            returns[t] = running
        return returns

    def _update_episode(self, obs_list, action_list, rewards) -> dict:
        returns = self._discounted_returns(rewards)
        obs_t = to_tensor(np.asarray(obs_list, dtype=np.float32), self.device)
        returns_t = to_tensor(returns, self.device)

        dist = self.actor(obs_t)
        if self.discrete:
            actions_t = to_tensor(np.asarray(action_list), self.device, dtype=torch.long)
            log_probs = dist.log_prob(actions_t)
            entropy = dist.entropy().mean()
        else:
            actions_t = to_tensor(np.asarray(action_list, dtype=np.float32), self.device)
            log_probs = dist.log_prob(actions_t).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()

        if self.baseline:
            values = self.critic(obs_t)
            advantages = returns_t - values.detach()
            critic_loss = ((returns_t - values) ** 2).mean()
        else:
            advantages = returns_t
            critic_loss = torch.zeros((), device=self.device)

        if self.normalize_returns and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_loss = -(advantages * log_probs).mean() - self.ent_coef * entropy
        loss = policy_loss + 0.5 * critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"policy_loss": float(policy_loss.item()), "value_loss": float(critic_loss.item())}

    def learn(self, total_steps: int, callback=None, log_interval: int = 10) -> "REINFORCE":
        if callback is not None:
            callback.on_training_start(self)
        obs, _ = self.env.reset(seed=self.seed)
        obs_list, action_list, rewards = [], [], []
        returns_window: deque = deque(maxlen=100)
        episodes = 0

        while self.num_timesteps < total_steps:
            action = self.predict(obs, deterministic=False)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            obs_list.append(np.asarray(obs, dtype=np.float32))
            action_list.append(action)
            rewards.append(reward)
            obs = next_obs
            self.num_timesteps += 1

            if callback is not None and not callback.on_step():
                break

            if terminated or truncated:
                metrics = self._update_episode(obs_list, action_list, rewards)
                returns_window.append(float(np.sum(rewards)))
                episodes += 1
                if episodes % log_interval == 0:
                    self.logger.record("rollout/ep_return_mean", float(np.mean(returns_window)))
                    for k, v in metrics.items():
                        self.logger.record(f"train/{k}", v)
                    self.logger.dump(self.num_timesteps)
                obs, _ = self.env.reset()
                obs_list, action_list, rewards = [], [], []

        if callback is not None:
            callback.on_training_end()
        return self

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict() if self.critic is not None else None,
                "config": dict(
                    gamma=self.gamma, hidden_sizes=self.hidden_sizes,
                    baseline=self.baseline, ent_coef=self.ent_coef,
                    normalize_returns=self.normalize_returns,
                ),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, env: Env = None, device: str = "auto", **kwargs) -> "REINFORCE":
        checkpoint = torch.load(path, map_location=get_device(device), weights_only=False)
        agent = cls(env, device=device, **{**checkpoint["config"], **kwargs})
        agent.actor.load_state_dict(checkpoint["actor"])
        if agent.critic is not None and checkpoint["critic"] is not None:
            agent.critic.load_state_dict(checkpoint["critic"])
        return agent
