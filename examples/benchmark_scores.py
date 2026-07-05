"""Reproduce benchmark scores for every algorithm and write a Markdown table.

Runs each algorithm on a canonical task with a fixed seed, evaluates the final
(deterministic) policy, and writes the results to ``docs/benchmarks.md``. This is
the script behind the benchmark table in the docs - re-run it to reproduce.

Run: python examples/benchmark_scores.py
"""

from __future__ import annotations

import os
import time
import traceback

import numpy as np

from reinforce.algorithms import (
    A2C,
    C51,
    DDPG,
    DQN,
    IQL,
    PPO,
    QRDQN,
    REINFORCE,
    SAC,
    SARSA,
    TD3,
    TD3BC,
    ExpectedSARSA,
    QLearning,
    SACDiscrete,
)
from reinforce.data import collect_dataset
from reinforce.envs import CartPole, GridWorld, Pendulum, PointMass
from reinforce.training import evaluate_policy
from reinforce.utils import Logger, set_seed

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "benchmarks.md")
Q = Logger(verbose=0)
SEED = 0
results: list = []


def cartpole():
    return CartPole(max_steps=500)


def gridworld():
    return GridWorld(rows=4, cols=4)


def gridworld_oh():
    return GridWorld(rows=4, cols=4, one_hot=True)


def random_score(env_fn, episodes=20):
    rs = []
    for ep in range(episodes):
        env = env_fn()
        obs, _ = env.reset(seed=1000 + ep)
        done, tot = False, 0.0
        while not done:
            obs, r, term, trunc, _ = env.step(env.action_space.sample())
            tot += r
            done = term or trunc
        rs.append(tot)
    return float(np.mean(rs))


def run(name, env_name, env_fn, steps, build, eval_episodes=20, offline=False):
    print(f"  running {name} on {env_name} ({steps} steps) ...", flush=True)
    t0 = time.time()
    try:
        set_seed(SEED)
        agent = build()
        if offline:
            rng = np.random.default_rng(0)

            def behavior(o):
                return np.clip(-np.asarray(o) * 3 + 0.3 * rng.standard_normal(2), -1, 1).astype(np.float32)

            ds = collect_dataset(env_fn(), behavior, n_transitions=20_000, gamma=0.99, seed=0)
            agent.learn_offline(ds, total_steps=steps)
        else:
            agent.learn(steps)
        mean, std = evaluate_policy(agent, env_fn(), n_episodes=eval_episodes, seed=1)
        rnd = random_score(env_fn, episodes=eval_episodes)
        results.append(dict(algo=name, env=env_name, steps=steps, mean=mean, std=std,
                            random=rnd, time=time.time() - t0))
        print(f"    -> {mean:.2f} +/- {std:.2f} (random {rnd:.2f}) in {time.time()-t0:.0f}s", flush=True)
    except Exception:  # pragma: no cover - benchmark robustness
        print(f"    !! {name} failed:\n{traceback.format_exc()}", flush=True)
        results.append(dict(algo=name, env=env_name, steps=steps, mean=float("nan"),
                            std=float("nan"), random=float("nan"), time=time.time() - t0))


def main() -> None:
    t_all = time.time()

    # Tabular (GridWorld: optimal return ~0.95)
    for cls in (QLearning, SARSA, ExpectedSARSA):
        run(cls.__name__, "GridWorld", gridworld, 30_000,
            lambda cls=cls: cls(gridworld(), learning_rate=0.2, seed=SEED, logger=Q))

    # Value-based (CartPole, max return 500)
    run("DQN", "CartPole", cartpole, 40_000,
        lambda: DQN(cartpole(), learning_rate=1e-3, buffer_size=50_000, learning_starts=1000,
                    target_update_interval=500, exploration_fraction=0.2, seed=SEED, logger=Q))
    run("C51", "CartPole", cartpole, 40_000,
        lambda: C51(cartpole(), v_min=0.0, v_max=500.0, n_atoms=51, learning_rate=1e-3,
                    buffer_size=50_000, learning_starts=1000, target_update_interval=500,
                    exploration_fraction=0.2, seed=SEED, logger=Q))
    run("QRDQN", "CartPole", cartpole, 40_000,
        lambda: QRDQN(cartpole(), n_quantiles=51, learning_rate=1e-3, buffer_size=50_000,
                      learning_starts=1000, target_update_interval=500, exploration_fraction=0.2,
                      seed=SEED, logger=Q))
    run("SACDiscrete", "CartPole", cartpole, 15_000,
        lambda: SACDiscrete(cartpole(), learning_rate=3e-4, learning_starts=1000, batch_size=64,
                            buffer_size=50_000, hidden_sizes=(128, 128), tau=0.01, seed=SEED, logger=Q))

    # Policy gradient / actor-critic (CartPole)
    run("PPO", "CartPole", cartpole, 40_000,
        lambda: PPO(cartpole(), n_steps=1024, batch_size=64, n_epochs=10, seed=SEED, logger=Q))
    run("A2C", "CartPole", cartpole, 40_000,
        lambda: A2C(cartpole(), n_steps=16, seed=SEED, logger=Q))
    run("REINFORCE", "CartPole", cartpole, 30_000,
        lambda: REINFORCE(cartpole(), learning_rate=1e-3, seed=SEED, logger=Q))

    # Continuous control (Pendulum, higher = better, optimal ~ -150)
    run("DDPG", "Pendulum", Pendulum, 15_000,
        lambda: DDPG(Pendulum(), learning_starts=1000, batch_size=256, seed=SEED, logger=Q),
        eval_episodes=10)
    run("TD3", "Pendulum", Pendulum, 15_000,
        lambda: TD3(Pendulum(), learning_starts=1000, batch_size=256, seed=SEED, logger=Q),
        eval_episodes=10)
    run("SAC", "Pendulum", Pendulum, 15_000,
        lambda: SAC(Pendulum(), learning_starts=1000, batch_size=256, seed=SEED, logger=Q),
        eval_episodes=10)

    # Offline (PointMass, from a scripted-behaviour dataset; random ~ -42)
    run("TD3BC", "PointMass", PointMass, 10_000,
        lambda: TD3BC(PointMass(), alpha=2.5, batch_size=256, seed=SEED, logger=Q), offline=True)
    run("IQL", "PointMass", PointMass, 10_000,
        lambda: IQL(PointMass(), batch_size=256, expectile=0.7, beta=3.0, seed=SEED, logger=Q),
        offline=True)

    _write_table(time.time() - t_all)


def _write_table(total_time: float) -> None:
    lines = [
        "# Benchmarks",
        "",
        "Reproduced scores from "
        "[`examples/benchmark_scores.py`]"
        "(https://github.com/DenisDrobyshev/reinforce/blob/main/examples/benchmark_scores.py),",
        "single seed (0), CPU. Return is the mean +/- std of the final deterministic policy over",
        "evaluation episodes; **random** is a uniform-random policy on the same task for reference.",
        "",
        "| Algorithm | Environment | Steps | Return (mean ± std) | Random | Time (s) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in results:
        ret = "n/a" if np.isnan(r["mean"]) else f"{r['mean']:.2f} ± {r['std']:.2f}"
        rnd = "n/a" if np.isnan(r["random"]) else f"{r['random']:.2f}"
        lines.append(
            f"| {r['algo']} | {r['env']} | {r['steps']:,} | {ret} | {rnd} | {r['time']:.0f} |"
        )
    lines += [
        "",
        f"_Total wall-clock: {total_time:.0f}s. GridWorld optimal ≈ 0.95; CartPole max = 500; "
        "Pendulum optimal ≈ -150 (higher is better); PointMass random ≈ -42._",
        "",
    ]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
