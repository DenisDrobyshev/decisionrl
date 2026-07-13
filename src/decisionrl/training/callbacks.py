"""Lightweight training callbacks.

A callback's :meth:`on_step` is invoked once per environment step during
:meth:`learn`. Returning ``False`` stops training early (e.g. once a reward
threshold is reached).
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from ..core.env import Env
from .evaluate import evaluate_policy

__all__ = [
    "Callback",
    "CallbackList",
    "EvalCallback",
    "StopOnRewardThreshold",
    "CheckpointCallback",
    "ProgressBarCallback",
]


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
        # run every callback each step (no short-circuit), then stop if any asked to
        results = [cb.on_step() for cb in self.callbacks]
        return all(results)

    def on_training_end(self) -> None:
        for cb in self.callbacks:
            cb.on_training_end()


class EvalCallback(Callback):
    """Periodically evaluate the agent, track the best mean return, and
    optionally save the best model so far to ``best_model_save_path``.
    """

    def __init__(
        self,
        eval_env: Env,
        eval_freq: int = 1000,
        n_eval_episodes: int = 5,
        deterministic: bool = True,
        best_model_save_path: Optional[str] = None,
        verbose: int = 1,
    ) -> None:
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.deterministic = deterministic
        self.best_model_save_path = best_model_save_path
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
            if mean_r > self.best_mean_reward:
                self.best_mean_reward = mean_r
                if self.best_model_save_path is not None:
                    os.makedirs(os.path.dirname(os.path.abspath(self.best_model_save_path)), exist_ok=True)
                    self.agent.save(self.best_model_save_path)
                    if self.verbose:
                        print(f"[eval] new best {mean_r:.2f} -> saved {self.best_model_save_path}")
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


class CheckpointCallback(Callback):
    """Save the agent every ``save_freq`` steps to ``save_dir``."""

    def __init__(self, save_freq: int, save_dir: str, name_prefix: str = "model", verbose: int = 1) -> None:
        self.save_freq = int(save_freq)
        self.save_dir = save_dir
        self.name_prefix = name_prefix
        self.verbose = verbose

    def on_training_start(self, agent) -> None:
        self.agent = agent
        os.makedirs(self.save_dir, exist_ok=True)

    def on_step(self) -> bool:
        step = self.agent.num_timesteps
        if self.save_freq > 0 and step % self.save_freq == 0:
            path = os.path.join(self.save_dir, f"{self.name_prefix}_{step}.pt")
            self.agent.save(path)
            if self.verbose:
                print(f"[checkpoint] step={step} -> {path}")
        return True


class ProgressBarCallback(Callback):
    """Show a live tqdm progress bar (steps/s, ETA, recent return).

    tqdm is optional; without it, this callback is a silent no-op.
    """

    def __init__(self) -> None:
        self.pbar = None
        self._last = 0

    def on_training_start(self, agent) -> None:
        self.agent = agent
        total = getattr(agent, "_total_timesteps", None)
        self._last = agent.num_timesteps
        try:
            from tqdm.auto import tqdm

            self.pbar = tqdm(total=total, initial=self._last, unit="step", dynamic_ncols=True)
        except ImportError:
            print("[decisionrl] tqdm not installed; run `pip install tqdm` for a progress bar.")
            self.pbar = None

    def on_step(self) -> bool:
        if self.pbar is None:
            return True
        now = self.agent.num_timesteps
        self.pbar.update(now - self._last)
        self._last = now
        window = getattr(self.agent, "ep_return_buffer", None)
        if window:
            self.pbar.set_postfix(ep_return=f"{np.mean(window):.1f}", refresh=False)
        return True

    def on_training_end(self) -> None:
        if self.pbar is not None:
            self.pbar.close()
