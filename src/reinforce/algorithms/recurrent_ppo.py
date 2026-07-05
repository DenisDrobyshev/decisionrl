"""Recurrent PPO (LSTM) for partially observable tasks (discrete actions).

An LSTM carries information across timesteps so the agent can act on histories
rather than single observations. Hidden state is reset at episode boundaries
(via done-masking) both during rollout collection and during the truncated-BPTT
update, which minibatches over *environments* to keep each sequence intact.
Structure follows CleanRL's ``ppo_lstm``.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from ..core.agent import BaseAgent
from ..core.env import Env
from ..core.spaces import is_discrete
from ..networks.mlp import layer_init
from ..utils.torch_utils import explained_variance, get_device, to_tensor
from ..wrappers.vector import SyncVectorEnv

__all__ = ["RecurrentPPO"]


class _RecurrentAC(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, lstm_size: int = 128, encoder_size: int = 64) -> None:
        super().__init__()
        self.lstm_size = lstm_size
        self.encoder = nn.Sequential(layer_init(nn.Linear(obs_dim, encoder_size)), nn.Tanh())
        self.lstm = nn.LSTM(encoder_size, lstm_size)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.actor = layer_init(nn.Linear(lstm_size, n_actions), gain=0.01)
        self.critic = layer_init(nn.Linear(lstm_size, 1), gain=1.0)

    def initial_state(self, batch: int, device) -> tuple:
        return (
            torch.zeros(1, batch, self.lstm_size, device=device),
            torch.zeros(1, batch, self.lstm_size, device=device),
        )

    def get_states(self, x: torch.Tensor, lstm_state: tuple, done: torch.Tensor):
        hidden = self.encoder(x)
        batch = lstm_state[0].shape[1]
        hidden = hidden.reshape(-1, batch, hidden.shape[-1])
        done = done.reshape(-1, batch)
        outputs = []
        for h_t, d_t in zip(hidden, done):
            mask = (1.0 - d_t).view(1, -1, 1)
            out, lstm_state = self.lstm(h_t.unsqueeze(0), (mask * lstm_state[0], mask * lstm_state[1]))
            outputs.append(out)
        new_hidden = torch.flatten(torch.cat(outputs), 0, 1)
        return new_hidden, lstm_state

    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden).flatten()

    def get_action_and_value(self, x, lstm_state, done, action=None):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        logits = self.actor(hidden)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(hidden).flatten(), lstm_state


class RecurrentPPO(BaseAgent):
    def __init__(
        self,
        env: Env,
        learning_rate: float = 3e-4,
        n_steps: int = 128,
        n_epochs: int = 4,
        n_minibatches: int = 1,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        lstm_size: int = 128,
        anneal_lr: bool = False,
        device: str = "auto",
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        if hasattr(env, "num_envs"):
            self.venv = env
            self.num_envs = int(env.num_envs)
            obs_space, act_space = env.single_observation_space, env.single_action_space
        else:
            self.venv = SyncVectorEnv([lambda e=env: e])  # type: ignore[misc]
            self.num_envs = 1
            obs_space, act_space = env.observation_space, env.action_space

        super().__init__(_SpaceHolder(obs_space, act_space), seed=seed, **kwargs)
        self.env = env
        assert is_discrete(act_space), "RecurrentPPO currently supports discrete action spaces"

        self.device = get_device(device)
        self.n_steps = int(n_steps)
        self.n_epochs = int(n_epochs)
        self.n_minibatches = int(n_minibatches)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_range = float(clip_range)
        self.ent_coef = float(ent_coef)
        self.vf_coef = float(vf_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.lstm_size = int(lstm_size)
        self.anneal_lr = bool(anneal_lr)

        self.obs_dim = int(np.prod(obs_space.shape))
        self.n_actions = int(act_space.n)
        self.net = _RecurrentAC(self.obs_dim, self.n_actions, lstm_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        self.initial_lr = float(learning_rate)

        self.ep_return_buffer: deque = deque(maxlen=100)
        self._eval_state: Optional[tuple] = None

    # -- evaluation-time recurrence ---------------------------------------
    def reset_states(self) -> None:
        self._eval_state = self.net.initial_state(1, self.device)

    @torch.no_grad()
    def predict(self, obs, deterministic: bool = True) -> int:
        if self._eval_state is None:
            self.reset_states()
        obs_t = to_tensor(np.asarray(obs, dtype=np.float32).reshape(1, -1), self.device)
        hidden, self._eval_state = self.net.get_states(obs_t, self._eval_state, torch.zeros(1, device=self.device))
        logits = self.net.actor(hidden)
        action = logits.argmax(dim=-1) if deterministic else Categorical(logits=logits).sample()
        return int(action.item())

    def learn(self, total_steps: int, callback=None, log_interval: int = 1) -> "RecurrentPPO":
        self._total_timesteps = total_steps
        if callback is not None:
            callback.on_training_start(self)
        device = self.device
        n, ne = self.n_steps, self.num_envs

        obs = torch.zeros((n, ne, self.obs_dim), device=device)
        actions = torch.zeros((n, ne), dtype=torch.long, device=device)
        logprobs = torch.zeros((n, ne), device=device)
        rewards = torch.zeros((n, ne), device=device)
        dones = torch.zeros((n, ne), device=device)
        values = torch.zeros((n, ne), device=device)

        next_obs = to_tensor(np.asarray(self.venv.reset(seed=self.seed)[0], dtype=np.float32), device)
        next_done = torch.zeros(ne, device=device)
        next_lstm_state = self.net.initial_state(ne, device)
        ep_returns = np.zeros(ne, dtype=np.float32)

        envsperbatch = max(1, ne // self.n_minibatches)

        while self.num_timesteps < total_steps:
            if self.anneal_lr:
                frac = max(0.0, 1.0 - self.num_timesteps / total_steps)
                for g in self.optimizer.param_groups:
                    g["lr"] = frac * self.initial_lr

            initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())

            for step in range(n):
                obs[step] = next_obs
                dones[step] = next_done
                with torch.no_grad():
                    action, logprob, _, value, next_lstm_state = self.net.get_action_and_value(
                        next_obs, next_lstm_state, next_done
                    )
                values[step] = value
                actions[step] = action
                logprobs[step] = logprob

                step_obs, reward, terminated, truncated, _ = self.venv.step(action.cpu().numpy())
                done = np.logical_or(terminated, truncated)
                ep_returns += np.asarray(reward, dtype=np.float32)
                for i in range(ne):
                    if done[i]:
                        self.ep_return_buffer.append(float(ep_returns[i]))
                        ep_returns[i] = 0.0

                rewards[step] = to_tensor(reward, device)
                next_obs = to_tensor(np.asarray(step_obs, dtype=np.float32), device)
                next_done = to_tensor(done.astype(np.float32), device)
                self.num_timesteps += ne

                if callback is not None and not callback.on_step():
                    if callback is not None:
                        callback.on_training_end()
                    return self

            # GAE
            with torch.no_grad():
                next_value = self.net.get_value(next_obs, next_lstm_state, next_done)
                advantages = torch.zeros_like(rewards)
                last_gae = torch.zeros(ne, device=device)
                for t in reversed(range(n)):
                    if t == n - 1:
                        next_nonterminal = 1.0 - next_done
                        next_val = next_value
                    else:
                        next_nonterminal = 1.0 - dones[t + 1]
                        next_val = values[t + 1]
                    delta = rewards[t] + self.gamma * next_val * next_nonterminal - values[t]
                    last_gae = delta + self.gamma * self.gae_lambda * next_nonterminal * last_gae
                    advantages[t] = last_gae
                returns = advantages + values

            metrics = self._update(obs, actions, logprobs, advantages, returns, values,
                                   dones, initial_lstm_state, envsperbatch)

            if log_interval and self.ep_return_buffer:
                self.logger.record("rollout/ep_return_mean", float(np.mean(self.ep_return_buffer)))
                for k, v in metrics.items():
                    self.logger.record(f"train/{k}", v)
                self.logger.dump(self.num_timesteps)

        if callback is not None:
            callback.on_training_end()
        return self

    def _update(self, obs, actions, logprobs, advantages, returns, values, dones,
                initial_lstm_state, envsperbatch) -> dict:
        n, ne = self.n_steps, self.num_envs
        b_obs = obs.reshape(-1, self.obs_dim)
        b_actions = actions.reshape(-1)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_dones = dones.reshape(-1)
        flatinds = np.arange(n * ne).reshape(n, ne)
        env_inds = np.arange(ne)

        pg_losses, vf_losses, entropies = [], [], []
        for _ in range(self.n_epochs):
            self.rng.shuffle(env_inds)
            for start in range(0, ne, envsperbatch):
                mb_envs = env_inds[start : start + envsperbatch]
                mb_inds = flatinds[:, mb_envs].ravel()
                lstm0 = (initial_lstm_state[0][:, mb_envs], initial_lstm_state[1][:, mb_envs])

                _, newlogprob, entropy, newvalue, _ = self.net.get_action_and_value(
                    b_obs[mb_inds], lstm0, b_dones[mb_inds], b_actions[mb_inds]
                )
                adv = b_advantages[mb_inds]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                ratio = torch.exp(newlogprob - b_logprobs[mb_inds])
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * adv
                pg_loss = -torch.min(surr1, surr2).mean()
                v_loss = ((newvalue - b_returns[mb_inds]) ** 2).mean()
                ent = entropy.mean()
                loss = pg_loss + self.vf_coef * v_loss - self.ent_coef * ent

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()
                pg_losses.append(pg_loss.item())
                vf_losses.append(v_loss.item())
                entropies.append(ent.item())

        ev = explained_variance(values.reshape(-1).cpu().numpy(), returns.reshape(-1).cpu().numpy())
        return {
            "policy_loss": float(np.mean(pg_losses)),
            "value_loss": float(np.mean(vf_losses)),
            "entropy": float(np.mean(entropies)),
            "explained_variance": float(ev),
        }

    def save(self, path: str) -> None:
        torch.save(
            {
                "net": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": dict(n_steps=self.n_steps, n_epochs=self.n_epochs,
                               n_minibatches=self.n_minibatches, gamma=self.gamma,
                               gae_lambda=self.gae_lambda, clip_range=self.clip_range,
                               ent_coef=self.ent_coef, vf_coef=self.vf_coef,
                               max_grad_norm=self.max_grad_norm, lstm_size=self.lstm_size),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, env: Env = None, device: str = "auto", **kwargs) -> "RecurrentPPO":
        checkpoint = torch.load(path, map_location=get_device(device), weights_only=False)
        agent = cls(env, device=device, **{**checkpoint["config"], **kwargs})
        agent.net.load_state_dict(checkpoint["net"])
        agent.optimizer.load_state_dict(checkpoint["optimizer"])
        return agent


class _SpaceHolder:
    def __init__(self, observation_space, action_space) -> None:
        self.observation_space = observation_space
        self.action_space = action_space
