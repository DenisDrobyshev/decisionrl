"""Meta-reinforcement learning via RL^2 (Duan et al., 2016; Wang et al., 2016).

RL^2 turns a *recurrent* policy into a fast learning algorithm: it is trained
across a distribution of tasks so that its hidden state, carried across episodes
within a single "trial", performs online adaptation. No gradient steps happen at
test time -- the recurrent dynamics *are* the learning algorithm.

The trick is entirely in the environment. :class:`RL2Env` samples a fresh task
each trial, then feeds the policy the previous action, reward and termination
flag alongside each observation, and keeps the same task alive (auto-resetting
its inner episodes) for a whole trial. Crucially the recurrent state is *not*
reset at inner-episode boundaries -- only between trials -- so experience
accumulates. Train any recurrent agent on it::

    from decisionrl.algorithms import RecurrentPPO
    from decisionrl.meta import make_meta_bandit

    env = make_meta_bandit(n_arms=5, horizon=50, seed=0)
    agent = RecurrentPPO(env, n_steps=50, seed=0).learn(200_000)

The learned policy explores the arms early in a trial and exploits the best one
later -- a bandit algorithm discovered by gradient descent.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

from .core.env import Env, Wrapper
from .core.spaces import Box, flatdim, flatten, is_discrete
from .envs.bandit import BernoulliBandit

__all__ = ["RL2Env", "make_meta_bandit"]

TaskFn = Callable[[np.random.Generator], Env]


class RL2Env(Wrapper):
    """Wrap a task distribution into a single meta-episode ("trial") environment.

    ``task_fn(rng)`` returns a freshly sampled task (an :class:`Env` with a
    discrete action space). Each call to :meth:`reset` starts a new trial by
    sampling a new task; the trial lasts ``horizon`` environment steps, during
    which inner-episode terminations auto-reset the *same* task and are surfaced
    to the agent through the observation instead of ending the trial. The
    observation is the task observation concatenated with the previous action
    (one-hot), the previous reward and the previous done flag.
    """

    def __init__(self, task_fn: TaskFn, horizon: int, seed: Optional[int] = None) -> None:
        self._task_fn = task_fn
        self._horizon = int(horizon)
        self._rng = np.random.default_rng(seed)

        probe = task_fn(np.random.default_rng(0))
        assert is_discrete(probe.action_space), "RL2Env requires a discrete action space"
        super().__init__(probe)
        self._n_actions = int(probe.action_space.n)
        self._base_dim = flatdim(probe.observation_space)
        self._base_space = probe.observation_space
        obs_dim = self._base_dim + self._n_actions + 2  # + prev reward + prev done
        self.observation_space = Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = probe.action_space

        self._steps = 0
        self._last_action = -1
        self._last_reward = 0.0
        self._last_done = 0.0

    def _augment(self, base_obs) -> np.ndarray:
        onehot = np.zeros(self._n_actions, dtype=np.float32)
        if self._last_action >= 0:
            onehot[self._last_action] = 1.0
        base = flatten(self._base_space, base_obs)
        return np.concatenate(
            [base, onehot, np.float32([self._last_reward, self._last_done])]
        ).astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.env = self._task_fn(self._rng)
        base_obs, info = self.env.reset()
        self._steps = 0
        self._last_action = -1
        self._last_reward = 0.0
        self._last_done = 0.0
        return self._augment(base_obs), info

    def step(self, action):
        base_obs, reward, terminated, truncated, info = self.env.step(action)
        self._steps += 1
        self._last_action = int(action)
        self._last_reward = float(reward)
        inner_done = bool(terminated or truncated)
        self._last_done = 1.0 if inner_done else 0.0

        trial_done = self._steps >= self._horizon
        if inner_done and not trial_done:
            base_obs, _ = self.env.reset()  # same task, fresh inner episode
        # The whole trial is one episode; expose its end as a truncation so the
        # agent bootstraps correctly and the recurrent state resets only here.
        return self._augment(base_obs), reward, False, trial_done, info


def make_meta_bandit(n_arms: int = 5, horizon: int = 50, seed: Optional[int] = None) -> RL2Env:
    """RL^2 meta-environment over Bernoulli bandits with resampled arm odds.

    Each trial draws fresh arm probabilities ``p_i ~ U(0, 1)`` and lasts
    ``horizon`` pulls. A meta-trained recurrent policy learns to explore then
    exploit within a single trial.
    """

    def task_fn(rng: np.random.Generator) -> Env:
        probs = rng.uniform(0.0, 1.0, size=int(n_arms))
        return BernoulliBandit(probs=probs, seed=int(rng.integers(2**31)))

    return RL2Env(task_fn, horizon=horizon, seed=seed)
