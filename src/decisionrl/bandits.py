"""Contextual bandit algorithms: LinUCB, linear Thompson sampling, epsilon-greedy.

These are the workhorses of operational decision-making where each choice is one-shot
and feedback is immediate: pricing, recommendation, ad and content selection, treatment
allocation. Unlike the reinforcement-learning agents elsewhere in the library, they are
closed-form linear methods with no gradient training, so they are fast, deterministic
under a seed, and cheap to reason about.

All three assume a linear reward model ``E[r | x, a] = x . theta_a`` and share the same
per-arm sufficient statistics (a ridge-regression design matrix and response vector).
They differ only in how they turn those statistics into an action:

* :class:`LinUCB` adds an optimism bonus (an upper confidence bound) to the estimate.
* :class:`LinearThompsonSampling` samples a parameter from the posterior and acts greedily.
* :class:`EpsilonGreedyBandit` acts greedily but explores uniformly with probability epsilon.

Use :func:`run_bandit` to play one against a :class:`~decisionrl.envs.ContextualBandit`
and get its cumulative reward and (exact) cumulative regret.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

__all__ = [
    "ContextualBanditAgent",
    "LinUCB",
    "LinearThompsonSampling",
    "EpsilonGreedyBandit",
    "run_bandit",
]


class ContextualBanditAgent:
    """Base class holding per-arm ridge statistics ``A_a`` and ``b_a``.

    ``A_a`` starts at ``lambda_ * I`` (ridge prior) and accumulates ``x x^T``; ``b_a``
    accumulates ``r x``. The ridge point estimate is ``theta_a = A_a^{-1} b_a``.
    """

    def __init__(self, n_arms: int, n_features: int, lambda_: float = 1.0,
                 seed: Optional[int] = None) -> None:
        self.n_arms = int(n_arms)
        self.n_features = int(n_features)
        self.rng = np.random.default_rng(seed)
        self.A = np.stack([lambda_ * np.eye(self.n_features) for _ in range(self.n_arms)])
        self.b = np.zeros((self.n_arms, self.n_features))

    def _theta(self, arm: int) -> np.ndarray:
        return np.linalg.solve(self.A[arm], self.b[arm])

    def select(self, context: np.ndarray) -> int:
        raise NotImplementedError

    def update(self, context: np.ndarray, arm: int, reward: float) -> None:
        x = np.asarray(context, dtype=np.float64)
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x


class LinUCB(ContextualBanditAgent):
    """LinUCB (Li et al., 2010): pick the arm with the highest upper confidence bound."""

    def __init__(self, n_arms: int, n_features: int, alpha: float = 1.0,
                 lambda_: float = 1.0, seed: Optional[int] = None) -> None:
        super().__init__(n_arms, n_features, lambda_=lambda_, seed=seed)
        self.alpha = float(alpha)

    def select(self, context: np.ndarray) -> int:
        x = np.asarray(context, dtype=np.float64)
        scores = np.empty(self.n_arms)
        for a in range(self.n_arms):
            a_inv_x = np.linalg.solve(self.A[a], x)
            mean = float(self.b[a] @ a_inv_x)  # theta_a . x  ==  b_a^T A_a^{-1} x
            bonus = self.alpha * float(np.sqrt(max(x @ a_inv_x, 0.0)))
            scores[a] = mean + bonus
        return int(np.argmax(scores))


class LinearThompsonSampling(ContextualBanditAgent):
    """Linear Thompson sampling: sample theta_a from its Gaussian posterior, act greedily."""

    def __init__(self, n_arms: int, n_features: int, v: float = 0.25,
                 lambda_: float = 1.0, seed: Optional[int] = None) -> None:
        super().__init__(n_arms, n_features, lambda_=lambda_, seed=seed)
        self.v = float(v)

    def select(self, context: np.ndarray) -> int:
        x = np.asarray(context, dtype=np.float64)
        scores = np.empty(self.n_arms)
        for a in range(self.n_arms):
            a_inv = np.linalg.inv(self.A[a])
            theta_hat = a_inv @ self.b[a]
            theta = self.rng.multivariate_normal(theta_hat, self.v**2 * a_inv)
            scores[a] = float(theta @ x)
        return int(np.argmax(scores))


class EpsilonGreedyBandit(ContextualBanditAgent):
    """Epsilon-greedy over the ridge estimate: explore uniformly, otherwise act greedily."""

    def __init__(self, n_arms: int, n_features: int, epsilon: float = 0.1,
                 lambda_: float = 1.0, seed: Optional[int] = None) -> None:
        super().__init__(n_arms, n_features, lambda_=lambda_, seed=seed)
        self.epsilon = float(epsilon)

    def select(self, context: np.ndarray) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_arms))
        x = np.asarray(context, dtype=np.float64)
        scores = [float(self._theta(a) @ x) for a in range(self.n_arms)]
        return int(np.argmax(scores))


def run_bandit(agent: ContextualBanditAgent, env, seed: int = 0) -> Dict[str, object]:
    """Play ``agent`` on a contextual-bandit ``env`` for one episode.

    Returns cumulative reward, cumulative regret, and the per-round regret curve. Regret
    is exact because the environment reports the gap to the optimal arm each round.
    """
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    regrets: List[float] = []
    done = False
    while not done:
        arm = agent.select(obs)
        next_obs, reward, term, trunc, info = env.step(arm)
        agent.update(obs, arm, reward)
        total_reward += reward
        regrets.append(float(info["regret"]))
        obs, done = next_obs, term or trunc
    cumulative = np.cumsum(regrets)
    return {
        "cumulative_reward": total_reward,
        "cumulative_regret": float(cumulative[-1]) if len(cumulative) else 0.0,
        "regret_curve": cumulative,
    }
