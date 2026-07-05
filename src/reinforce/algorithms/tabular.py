"""Tabular temporal-difference control: Q-Learning, SARSA, Expected SARSA.

These require discrete observation *and* action spaces. They are exact (no
function approximation) and converge to the optimal policy on small MDPs, which
makes them ideal teaching baselines and correctness anchors for the rest of the
library.
"""

from __future__ import annotations

import pickle
from typing import Optional

import numpy as np

from ..core.agent import BaseAgent
from ..core.env import Env
from ..core.spaces import is_discrete
from ..exploration.schedules import LinearSchedule

__all__ = ["QLearning", "SARSA", "ExpectedSARSA"]


class _TabularBase(BaseAgent):
    def __init__(
        self,
        env: Env,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        exploration_fraction: float = 0.5,
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(env, seed=seed, **kwargs)
        assert is_discrete(self.observation_space), "tabular methods need a Discrete observation space"
        assert is_discrete(self.action_space), "tabular methods need a Discrete action space"
        self.lr = float(learning_rate)
        self.gamma = float(gamma)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.exploration_fraction = float(exploration_fraction)

        self.n_states = int(self.observation_space.n)
        self.n_actions = int(self.action_space.n)
        self.q_table = np.zeros((self.n_states, self.n_actions), dtype=np.float64)
        self.epsilon = epsilon_start

    def _epsilon_greedy(self, state: int, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
        # random tie-breaking among the best actions
        row = self.q_table[state]
        best = np.flatnonzero(row == row.max())
        return int(self.rng.choice(best))

    def predict(self, obs, deterministic: bool = True) -> int:
        state = int(obs)
        if deterministic:
            # Stable argmax so that deterministic prediction is reproducible.
            return int(np.argmax(self.q_table[state]))
        return self._epsilon_greedy(state, self.epsilon)

    # to be provided by subclasses ----------------------------------------
    def _td_target(self, reward: float, next_state: int, done: bool, epsilon: float) -> float:
        raise NotImplementedError

    def learn(self, total_steps: int, callback=None, log_interval: int = 100) -> "_TabularBase":
        schedule = LinearSchedule(
            self.epsilon_start,
            self.epsilon_end,
            max(1, int(self.exploration_fraction * total_steps)),
        )
        self._total_timesteps = self.num_timesteps + total_steps
        if callback is not None:
            callback.on_training_start(self)

        obs, _ = self.env.reset(seed=self.seed)
        state = int(obs)
        ep_return, ep_len, episodes = 0.0, 0, 0
        returns_window: list = []

        for step in range(total_steps):
            self.epsilon = schedule(step)
            action = self._epsilon_greedy(state, self.epsilon)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            next_state = int(next_obs)
            done = terminated  # bootstrap flag: truncation still bootstraps

            target = self._td_target(reward, next_state, done, self.epsilon)
            self.q_table[state, action] += self.lr * (target - self.q_table[state, action])

            state = next_state
            ep_return += reward
            ep_len += 1
            self.num_timesteps += 1

            if callback is not None and not callback.on_step():
                break

            if terminated or truncated:
                episodes += 1
                returns_window.append(ep_return)
                returns_window = returns_window[-100:]
                if episodes % log_interval == 0:
                    self.logger.record("rollout/ep_return_mean", float(np.mean(returns_window)))
                    self.logger.record("rollout/epsilon", self.epsilon)
                    self.logger.dump(self.num_timesteps)
                obs, _ = self.env.reset()
                state = int(obs)
                ep_return, ep_len = 0.0, 0

        if callback is not None:
            callback.on_training_end()
        return self

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "q_table": self.q_table,
                    "gamma": self.gamma,
                    "lr": self.lr,
                    "class": type(self).__name__,
                },
                f,
            )

    @classmethod
    def load(cls, path: str, env: Optional[Env] = None, **kwargs) -> "_TabularBase":
        with open(path, "rb") as f:
            data = pickle.load(f)
        agent = cls(env, learning_rate=data["lr"], gamma=data["gamma"], **kwargs)
        agent.q_table = data["q_table"]
        return agent


class QLearning(_TabularBase):
    """Off-policy TD control (Watkins, 1989): bootstrap on the greedy action."""

    def _td_target(self, reward, next_state, done, epsilon) -> float:
        if done:
            return reward
        return reward + self.gamma * float(self.q_table[next_state].max())


class SARSA(_TabularBase):
    """On-policy TD control: bootstrap on the epsilon-greedy next action."""

    def _td_target(self, reward, next_state, done, epsilon) -> float:
        if done:
            return reward
        next_action = self._epsilon_greedy(next_state, epsilon)
        return reward + self.gamma * float(self.q_table[next_state, next_action])


class ExpectedSARSA(_TabularBase):
    """Expected SARSA: bootstrap on the expected value under the behaviour policy."""

    def _td_target(self, reward, next_state, done, epsilon) -> float:
        if done:
            return reward
        row = self.q_table[next_state]
        best = np.flatnonzero(row == row.max())
        probs = np.full(self.n_actions, epsilon / self.n_actions)
        probs[best] += (1.0 - epsilon) / len(best)
        return reward + self.gamma * float(np.dot(probs, row))
