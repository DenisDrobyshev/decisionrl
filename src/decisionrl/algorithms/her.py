"""Hindsight Experience Replay (Andrychowicz et al., 2017) with a DQN backbone.

HER makes sparse-reward, goal-conditioned tasks learnable: every failed episode is
*relabelled* as if a goal it actually reached had been the target, turning sparse
failures into dense successes. This ``HERDQN`` stores whole episodes and, at sample
time, replaces the desired goal of a fraction of transitions with a **future**
achieved goal (the "future" strategy) and recomputes the reward.
"""

from __future__ import annotations

from collections import deque
from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..core.agent import BaseAgent
from ..core.env import Env
from ..exploration.schedules import LinearSchedule
from ..networks.mlp import build_mlp
from ..utils.torch_utils import get_device, hard_update, soft_update, to_tensor

__all__ = ["HERReplayBuffer", "HERDQN"]


class HERReplayBuffer:
    """Episode replay that relabels goals with future achieved goals ("future")."""

    def __init__(self, capacity: int, n_bits: int, compute_reward: Callable, her_ratio: float = 0.8,
                 seed: Optional[int] = None) -> None:
        self.capacity = int(capacity)
        self.n_bits = int(n_bits)
        self.compute_reward = compute_reward
        self.her_ratio = float(her_ratio)
        self.rng = np.random.default_rng(seed)
        self.episodes: deque = deque(maxlen=self.capacity)

    def store_episode(self, states, actions, next_states) -> None:
        self.episodes.append((np.asarray(states, np.float32), np.asarray(actions, np.int64),
                              np.asarray(next_states, np.float32)))

    def __len__(self) -> int:
        return sum(len(a) for _, a, _ in self.episodes)

    def sample(self, batch_size: int, device):
        obs, act, rew, next_obs, done = [], [], [], [], []
        for _ in range(batch_size):
            states, actions, next_states = self.episodes[self.rng.integers(len(self.episodes))]
            t = int(self.rng.integers(len(actions)))
            if self.rng.random() < self.her_ratio:  # relabel with a future achieved goal
                future = int(self.rng.integers(t, len(actions)))
                goal = next_states[future][: self.n_bits]
            else:  # keep the original desired goal (stored in the obs halves)
                goal = states[t][self.n_bits:]
            achieved = next_states[t][: self.n_bits]
            r = self.compute_reward(achieved, goal)
            obs.append(np.concatenate([states[t][: self.n_bits], goal]))
            next_obs.append(np.concatenate([next_states[t][: self.n_bits], goal]))
            act.append(actions[t])
            rew.append(r)
            done.append(1.0 if r == 0.0 else 0.0)
        f = lambda a, d=torch.float32: torch.as_tensor(np.asarray(a), device=device, dtype=d)  # noqa: E731
        return f(obs), f(act, torch.long), f(rew), f(next_obs), f(done)


class HERDQN(BaseAgent):
    def __init__(
        self,
        env: Env,
        learning_rate: float = 1e-3,
        gamma: float = 0.98,
        buffer_size: int = 100_000,
        batch_size: int = 128,
        hidden_sizes: Sequence[int] = (256, 256),
        her_ratio: float = 0.8,
        tau: float = 0.05,
        target_update_interval: int = 40,
        gradient_steps: int = 40,
        exploration_fraction: float = 0.2,
        epsilon_end: float = 0.05,
        learning_starts: int = 1,
        device: str = "auto",
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(env, seed=seed, **kwargs)
        self.device = get_device(device)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.tau = float(tau)
        self.target_update_interval = int(target_update_interval)
        self.gradient_steps = int(gradient_steps)
        self.n_bits = int(env.action_space.n)
        self.obs_dim = int(np.prod(self.observation_space.shape))
        self.n_actions = int(self.action_space.n)
        assert hasattr(env, "compute_reward"), "HERDQN requires a goal env with compute_reward()"

        self.q_net = build_mlp(self.obs_dim, self.n_actions, hidden_sizes).to(self.device)
        self.q_target = build_mlp(self.obs_dim, self.n_actions, hidden_sizes).to(self.device)
        hard_update(self.q_net, self.q_target)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.buffer = HERReplayBuffer(buffer_size, self.n_bits, env.compute_reward, her_ratio, seed)
        self.exploration_fraction = float(exploration_fraction)
        self.epsilon_end = float(epsilon_end)
        self.learning_starts = int(learning_starts)
        self._epsilon = 1.0

    @torch.no_grad()
    def predict(self, obs, deterministic: bool = True) -> int:
        if not deterministic and self.rng.random() < self._epsilon:
            return int(self.rng.integers(self.n_actions))
        obs_t = to_tensor(np.asarray(obs, np.float32).reshape(1, -1), self.device)
        return int(self.q_net(obs_t).argmax(dim=-1).item())

    def _update(self) -> float:
        losses = []
        for _ in range(self.gradient_steps):
            obs, act, rew, next_obs, done = self.buffer.sample(self.batch_size, self.device)
            with torch.no_grad():
                target = rew + self.gamma * (1 - done) * self.q_target(next_obs).max(dim=-1).values
            q = self.q_net(obs).gather(1, act.unsqueeze(1)).squeeze(1)
            loss = F.mse_loss(q, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.item()))
        return float(np.mean(losses))

    def learn(self, total_steps: int, callback=None, log_interval: int = 20) -> "HERDQN":
        sched = LinearSchedule(1.0, self.epsilon_end, int(self.exploration_fraction * total_steps))
        if callback is not None:
            callback.on_training_start(self)
        success: deque = deque(maxlen=100)
        episodes = 0
        while self.num_timesteps < total_steps:
            self._epsilon = sched(self.num_timesteps)
            obs, _ = self.env.reset()
            states, actions, next_states = [], [], []
            done, solved = False, False
            while not done:
                action = self.predict(obs, deterministic=False)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                states.append(np.asarray(obs, np.float32))
                actions.append(action)
                next_states.append(np.asarray(next_obs, np.float32))
                obs = next_obs
                self.num_timesteps += 1
                solved = solved or terminated
                done = terminated or truncated
            self.buffer.store_episode(states, actions, next_states)
            success.append(1.0 if solved else 0.0)
            episodes += 1

            loss = 0.0
            if episodes >= self.learning_starts:
                loss = self._update()
                soft_update(self.q_net, self.q_target, self.tau)
            if callback is not None and not callback.on_step():
                break
            if episodes % log_interval == 0:
                self.logger.record("rollout/success_rate", float(np.mean(success)))
                self.logger.record("train/loss", loss)
                self.logger.dump(self.num_timesteps)
        if callback is not None:
            callback.on_training_end()
        return self

    def success_rate(self, n_episodes: int = 50, seed: int = 123) -> float:
        wins = 0
        for ep in range(n_episodes):
            obs, _ = self.env.reset(seed=seed + ep)
            done = False
            while not done:
                obs, r, terminated, truncated, _ = self.env.step(self.predict(obs, deterministic=True))
                wins += int(terminated)
                done = terminated or truncated
        return wins / n_episodes

    def save(self, path: str) -> None:
        torch.save({"q_net": self.q_net.state_dict(),
                    "config": dict(gamma=self.gamma, batch_size=self.batch_size)}, path)

    @classmethod
    def load(cls, path: str, env: Env = None, device: str = "auto", **kwargs) -> "HERDQN":
        ckpt = torch.load(path, map_location=get_device(device), weights_only=False)
        agent = cls(env, device=device, **{**ckpt["config"], **kwargs})
        agent.q_net.load_state_dict(ckpt["q_net"])
        hard_update(agent.q_net, agent.q_target)
        return agent
