"""Reliable evaluation statistics for RL (rliable-style; Agarwal et al., 2021).

Point estimates from a handful of seeds are noisy and often misleading. This
module provides robust aggregate metrics with **stratified bootstrap confidence
intervals**, **performance profiles** and **probability of improvement**, plus a
small multi-seed runner, so results can be reported honestly.

    scores = run_seeds(lambda e: PPO(e, seed=0), CartPole, seeds=range(5), steps=50_000)
    print(aggregate_metrics(scores))          # mean / median / IQM, each with a 95% CI
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "iqm",
    "bootstrap_ci",
    "aggregate_metrics",
    "performance_profile",
    "probability_of_improvement",
    "run_seeds",
]


def iqm(scores: np.ndarray) -> float:
    """Interquartile mean: the mean of the middle 50% (robust to outliers)."""
    x = np.sort(np.asarray(scores, dtype=np.float64).ravel())
    n = x.size
    if n == 0:
        return float("nan")
    lo, hi = int(np.floor(n * 0.25)), int(np.ceil(n * 0.75))
    trimmed = x[lo:hi] if hi > lo else x
    return float(np.mean(trimmed))


def bootstrap_ci(
    scores: np.ndarray,
    aggregate: Callable[[np.ndarray], float] = iqm,
    reps: int = 2000,
    ci: float = 0.95,
    seed: Optional[int] = 0,
) -> Tuple[float, float, float]:
    """Return ``(point_estimate, ci_low, ci_high)`` via bootstrap resampling."""
    x = np.asarray(scores, dtype=np.float64).ravel()
    rng = np.random.default_rng(seed)
    point = aggregate(x)
    boot = np.array([aggregate(rng.choice(x, size=x.size, replace=True)) for _ in range(reps)])
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot, [alpha, 1 - alpha])
    return float(point), float(lo), float(hi)


def aggregate_metrics(scores: np.ndarray, reps: int = 2000, ci: float = 0.95,
                      seed: Optional[int] = 0) -> Dict[str, Tuple[float, float, float]]:
    """Mean, median and IQM of ``scores``, each with a bootstrap CI."""
    aggregates = {"mean": np.mean, "median": np.median, "iqm": iqm}
    return {name: bootstrap_ci(scores, agg, reps, ci, seed) for name, agg in aggregates.items()}


def performance_profile(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Fraction of runs with score >= tau, for each tau in ``thresholds``.

    The resulting run-score distribution / performance profile is monotone
    non-increasing in tau and lies in [0, 1].
    """
    x = np.asarray(scores, dtype=np.float64).ravel()
    taus = np.asarray(thresholds, dtype=np.float64)
    return np.array([float(np.mean(x >= t)) for t in taus])


def probability_of_improvement(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """P(A > B): probability a random A-run beats a random B-run (ties = 0.5)."""
    a = np.asarray(scores_a, dtype=np.float64).ravel()
    b = np.asarray(scores_b, dtype=np.float64).ravel()
    wins = (a[:, None] > b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum()
    return float(wins / (a.size * b.size))


def run_seeds(
    agent_fn: Callable,
    env_fn: Callable,
    seeds: Sequence[int],
    steps: int,
    eval_episodes: int = 10,
    eval_seed: int = 1000,
) -> np.ndarray:
    """Train one agent per seed and return the array of mean evaluation returns.

    ``agent_fn(env)`` builds an agent on a fresh ``env_fn()``; each is trained for
    ``steps`` and evaluated with :func:`~reinforce.training.evaluate_policy`.
    """
    from .training import evaluate_policy
    from .utils import set_seed

    results = []
    for s in seeds:
        set_seed(int(s))
        agent = agent_fn(env_fn())
        agent.learn(steps)
        mean_return, _ = evaluate_policy(agent, env_fn(), n_episodes=eval_episodes, seed=eval_seed)
        results.append(mean_return)
    return np.asarray(results, dtype=np.float64)
