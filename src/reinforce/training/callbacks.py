"""Lightweight training callbacks.

A callback's :meth:`on_step` is invoked once per environment step during
:meth:`learn`. Returning ``False`` stops training early (e.g. once a reward
threshold is reached).
"""

from __future__ import annotations

from typing import List

from ..core.env import Env
from .evaluate import evaluate_policy

__all__ = ["Callback", "CallbackList", "EvalCallback", "StopOnRewardThreshold"]


class Callback:
    def on_training_start(self, agent) -> None:
        self.agent = agent

    def on_step(self) -> bool:
        """Return ``False`` to stop training."""
        return True

    def on_training_end(self) -> None:
        pass


class CallbackList(Callback):
    def __init__(self, callbacks: List[Callback]) -> None:
        self.callbacks = list(callbacks)

    def on_training_start(self, agent) -> None:
        self.agent = agent
        for cb in self.callbacks:
            cb.on_training_start(agent)

    def on_step(self) -> bool:
        return all(cb.on_step() for cb in self.callbacks)

    def on_training_end(self) -> None:
        for cb in self.callbacks:
            cb.on_training_end()


class EvalCallback(Callback):
    """Periodically evaluate the agent and remember the best mean return."""

    def __init__(
        self,
        eval_env: Env,
        eval_freq: int = 1000,
        n_eval_episodes: int = 5,
        deterministic: bool = True,
        verbose: int = 1,
    ) -> None:
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.deterministic = deterministic
        self.verbose = verbose
        self.best_mean_reward = -float("inf")
        self.last_mean_reward = -float("inf")
        self.evaluations: List[float] = []

    def on_step(self) -> bool:
        step = self.agent.num_timesteps
        if self.eval_freq > 0 and step % self.eval_freq == 0:
            mean_r, std_r = evaluate_policy(
                self.agent, self.eval_env, self.n_eval_episodes, self.deterministic
            )
            self.last_mean_reward = mean_r
            self.evaluations.append(mean_r)
            self.best_mean_reward = max(self.best_mean_reward, mean_r)
            if self.verbose:
                print(f"[eval] step={step} mean_reward={mean_r:.2f} +/- {std_r:.2f}")
        return True


class StopOnRewardThreshold(Callback):
    """Stop training once evaluation mean reward exceeds a threshold."""

    def __init__(self, eval_callback: EvalCallback, threshold: float) -> None:
        self.eval_callback = eval_callback
        self.threshold = float(threshold)

    def on_step(self) -> bool:
        return self.eval_callback.last_mean_reward < self.threshold
