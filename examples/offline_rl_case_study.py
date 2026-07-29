"""Offline RL from logged operational data: learn a battery-control policy from logs.

Real operations rarely offer a live simulator to explore in; what you have is a log of
past decisions and their outcomes. This study mirrors that. A noisy greedy controller
runs the :class:`~decisionrl.envs.EnergyMicrogrid` battery and its transitions are
recorded as a fixed dataset. An offline agent (IQL or TD3+BC) is then trained on that log
alone, with no further environment interaction, and evaluated. The question is whether it
can match or improve on the behaviour policy that generated the data.

Run: python examples/offline_rl_case_study.py [--seeds 5] [--steps 50000]
     [--transitions 40000] [--algo iql|td3bc] [--device auto]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from decisionrl import baselines as B
from decisionrl.algorithms import IQL, TD3BC
from decisionrl.data import collect_dataset
from decisionrl.envs import EnergyMicrogrid
from decisionrl.evaluation import bootstrap_ci, iqm
from decisionrl.training import evaluate_policy
from decisionrl.utils import Logger, set_seed

ALGOS = {"iql": IQL, "td3bc": TD3BC}
_ATTR_ENV = EnergyMicrogrid()  # a fixed instance for the greedy policy's static attributes


def _noisy_greedy(noise: float, rng: np.random.Generator):
    """A greedy price-threshold battery controller with exploration noise, as (env, obs)."""
    threshold, _ = B.best_price_threshold_battery(EnergyMicrogrid, seed=100)
    greedy = B.price_threshold_battery(float(threshold))
    low = float(_ATTR_ENV.action_space.low[0])
    high = float(_ATTR_ENV.action_space.high[0])

    def policy(env, obs):
        action = np.asarray(greedy(env, obs), dtype=np.float32).reshape(-1)
        return np.clip(action + noise * rng.normal(size=action.shape), low, high).astype(np.float32)

    return policy


def behavior_policy_return(noise: float) -> float:
    """Return of the noisy behaviour policy that generates the logs (the offline baseline)."""
    policy = _noisy_greedy(noise, np.random.default_rng(123))
    return B.rollout_return(EnergyMicrogrid, policy, episodes=40, seed=100)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--transitions", type=int, default=40_000)
    ap.add_argument("--noise", type=float, default=0.3)
    ap.add_argument("--algo", choices=list(ALGOS), default="iql")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    behavior_value = behavior_policy_return(args.noise)
    print(f"behaviour policy (noisy greedy, noise={args.noise}): {behavior_value:.2f} return")
    print(f"logging {args.transitions} transitions from it, then training offline only...")

    t0 = time.time()
    scores = []
    for s in range(args.seeds):
        set_seed(s)
        logger_policy = _noisy_greedy(args.noise, np.random.default_rng(s))
        dataset = collect_dataset(
            EnergyMicrogrid(), lambda obs, p=logger_policy: p(_ATTR_ENV, obs),
            args.transitions, seed=s)
        agent = ALGOS[args.algo](EnergyMicrogrid(), seed=s, logger=Logger(verbose=0),
                                 device=args.device)
        agent.learn_offline(dataset, args.steps)
        score = evaluate_policy(agent, EnergyMicrogrid(), n_episodes=30, seed=100)[0]
        scores.append(score)
        print(f"  seed {s}: {score:.2f}")

    point, lo, hi = bootstrap_ci(scores, aggregate=iqm, reps=5000, seed=0)
    lift = 100.0 * (point - behavior_value) / abs(behavior_value)
    print(f"\noffline {args.algo.upper()} IQM {point:.2f} [95% CI {lo:.2f}, {hi:.2f}]  "
          f"vs behaviour policy {behavior_value:.2f}  ({lift:+.0f}%)")
    print(f"{args.seeds} seeds, {args.steps} offline steps each, no environment interaction, "
          f"{time.time() - t0:.0f}s")

    out = {"env": "EnergyMicrogrid", "algo": args.algo, "behavior_value": behavior_value,
           "offline_iqm": point, "ci_low": lo, "ci_high": hi, "rl_mean": statistics.mean(scores),
           "seed_values": scores, "seeds": args.seeds, "steps": args.steps,
           "transitions": args.transitions}
    Path("offline_rl_case_study.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
