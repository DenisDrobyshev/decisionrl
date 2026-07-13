"""Head-to-head benchmark: ``decisionrl`` vs Stable-Baselines3 (and CleanRL).

Trains matched algorithms on the *same* Gymnasium environments, over several
seeds and an identical step budget, then reports mean +/- std evaluation return
and wall-clock time side by side. Results are written to JSON so runs are
reproducible and comparable across machines.

The ``decisionrl`` side always runs (Gymnasium is the only requirement). The
Stable-Baselines3 side runs only if SB3 is installed (`pip install
stable_baselines3`); otherwise it is skipped with a note. CleanRL is a collection
of single-file scripts rather than a package, so it is compared by running its
reference scripts separately — see ``docs/benchmarks.md`` for the procedure.

Examples
--------
    # decisionrl vs SB3 on CartPole, 3 seeds, 50k steps each
    python examples/benchmark_vs_baselines.py --algos ppo --env CartPole-v1 \
        --seeds 3 --steps 50000

    # continuous control (needs `pip install "gymnasium[mujoco]"` + SB3)
    python examples/benchmark_vs_baselines.py --algos sac --env Pendulum-v1 \
        --seeds 3 --steps 30000
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path
from typing import Optional

import numpy as np

from decisionrl import make_agent, make_env
from decisionrl.training import evaluate_policy
from decisionrl.utils import set_seed

# decisionrl algo name -> Stable-Baselines3 class name (only algos both libraries have)
SB3_EQUIVALENT = {
    "ppo": "PPO",
    "a2c": "A2C",
    "dqn": "DQN",
    "sac": "SAC",
    "td3": "TD3",
    "ddpg": "DDPG",
}

SB3_AVAILABLE = importlib.util.find_spec("stable_baselines3") is not None


def run_reinforce(algo: str, env_id: str, steps: int, seed: int) -> dict:
    set_seed(seed)
    env = make_env(f"gym:{env_id}")
    agent = make_agent(algo, env, seed=seed)
    t0 = time.perf_counter()
    agent.learn(total_steps=steps)
    wall = time.perf_counter() - t0
    mean, std = evaluate_policy(agent, make_env(f"gym:{env_id}"), n_episodes=20, seed=seed)
    env.close()
    return {"mean": float(mean), "std": float(std), "wall_s": wall}


def run_sb3(algo: str, env_id: str, steps: int, seed: int) -> Optional[dict]:
    if not SB3_AVAILABLE:
        return None
    import stable_baselines3 as sb3
    from stable_baselines3.common.evaluation import evaluate_policy as sb3_eval

    cls = getattr(sb3, SB3_EQUIVALENT[algo])
    policy = "MlpPolicy"
    model = cls(policy, env_id, seed=seed, verbose=0)
    t0 = time.perf_counter()
    model.learn(total_timesteps=steps)
    wall = time.perf_counter() - t0
    mean, std = sb3_eval(model, model.get_env(), n_eval_episodes=20, deterministic=True)
    return {"mean": float(mean), "std": float(std), "wall_s": wall}


def aggregate(runs: list[dict]) -> dict:
    means = [r["mean"] for r in runs]
    walls = [r["wall_s"] for r in runs]
    return {
        "return_mean": statistics.mean(means),
        "return_std": statistics.pstdev(means) if len(means) > 1 else 0.0,
        "wall_s_mean": statistics.mean(walls),
        "n_seeds": len(runs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algos", nargs="+", default=["ppo"],
                        help=f"one or more of {sorted(SB3_EQUIVALENT)}")
    parser.add_argument("--env", default="CartPole-v1", help="Gymnasium env id")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--output", default="benchmark_results")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True)
    results: dict = {"env": args.env, "steps": args.steps, "seeds": args.seeds, "algos": {}}

    for algo in args.algos:
        if algo not in SB3_EQUIVALENT:
            raise SystemExit(f"unknown algo {algo!r}; choose from {sorted(SB3_EQUIVALENT)}")
        print(f"\n=== {algo.upper()} on {args.env} ({args.seeds} seeds x {args.steps} steps) ===")

        rl_runs, sb3_runs = [], []
        for seed in range(args.seeds):
            rl = run_reinforce(algo, args.env, args.steps, seed)
            rl_runs.append(rl)
            print(f"  decisionrl seed {seed}: {rl['mean']:.1f} +/- {rl['std']:.1f} "
                  f"({rl['wall_s']:.0f}s)")
            sb3 = run_sb3(algo, args.env, args.steps, seed)
            if sb3 is not None:
                sb3_runs.append(sb3)
                print(f"  SB3       seed {seed}: {sb3['mean']:.1f} +/- {sb3['std']:.1f} "
                      f"({sb3['wall_s']:.0f}s)")

        entry = {"decisionrl": aggregate(rl_runs)}
        if sb3_runs:
            entry["sb3"] = aggregate(sb3_runs)
        results["algos"][algo] = entry

    # markdown comparison table
    print("\n| Algo | Env | decisionrl (return) | SB3 (return) | decisionrl (s) | SB3 (s) |")
    print("|---|---|---|---|---|---|")
    for algo, entry in results["algos"].items():
        rl = entry["decisionrl"]
        sb3 = entry.get("sb3")
        rl_ret = f"{rl['return_mean']:.1f} ± {rl['return_std']:.1f}"
        rl_wall = f"{rl['wall_s_mean']:.0f}"
        sb3_ret = f"{sb3['return_mean']:.1f} ± {sb3['return_std']:.1f}" if sb3 else "— (not installed)"
        sb3_wall = f"{sb3['wall_s_mean']:.0f}" if sb3 else "—"
        print(f"| {algo.upper()} | {args.env} | {rl_ret} | {sb3_ret} | {rl_wall} | {sb3_wall} |")

    if not SB3_AVAILABLE:
        print("\nNote: Stable-Baselines3 is not installed, so only decisionrl was run.")
        print("      `pip install stable_baselines3` to fill in the comparison column.")

    out_path = out_dir / f"{args.env}_{'_'.join(args.algos)}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    # deterministic-ish across processes
    np.seterr(all="ignore")
    main()
