"""Contextual bandits on a linear reward problem: LinUCB vs Thompson vs epsilon-greedy.

Runs each algorithm on the same :class:`~decisionrl.envs.ContextualBandit` and reports
cumulative regret (lower is better), averaged over several seeds. Confidence-based
methods (LinUCB, Thompson) explore efficiently and reach far lower regret than
epsilon-greedy, and all three are far below a random policy.

Run: python examples/contextual_bandits.py [--rounds 3000] [--seeds 5]
"""

from __future__ import annotations

import argparse

import numpy as np

from decisionrl.bandits import (
    EpsilonGreedyBandit,
    LinearThompsonSampling,
    LinUCB,
    run_bandit,
)
from decisionrl.envs import ContextualBandit

N_ARMS, N_FEATURES = 6, 8


def mean_regret(agent_cls, rounds, seeds, **kw) -> float:
    out = []
    for s in range(seeds):
        env = ContextualBandit(n_arms=N_ARMS, n_features=N_FEATURES, horizon=rounds, seed=42)
        agent = agent_cls(N_ARMS, N_FEATURES, seed=s, **kw)
        out.append(run_bandit(agent, env, seed=s)["cumulative_regret"])
    return float(np.mean(out))


def random_regret(rounds: int, seed: int = 0) -> float:
    env = ContextualBandit(n_arms=N_ARMS, n_features=N_FEATURES, horizon=rounds, seed=42)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    total, done = 0.0, False
    while not done:
        _, _, term, trunc, info = env.step(int(rng.integers(N_ARMS)))
        total += info["regret"]
        done = term or trunc
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    print(f"linear contextual bandit: {N_ARMS} arms, {N_FEATURES} features, "
          f"{args.rounds} rounds, {args.seeds} seeds")
    print(f"\n{'algorithm':<22}{'cumulative regret':>18}")
    print("-" * 40)
    rows = [
        ("LinUCB", LinUCB, dict(alpha=1.0)),
        ("Thompson sampling", LinearThompsonSampling, dict(v=0.25)),
        ("epsilon-greedy (0.1)", EpsilonGreedyBandit, dict(epsilon=0.1)),
    ]
    for name, cls, kw in rows:
        print(f"{name:<22}{mean_regret(cls, args.rounds, args.seeds, **kw):>18.1f}")
    print(f"{'random':<22}{random_regret(args.rounds):>18.1f}")


if __name__ == "__main__":
    main()
