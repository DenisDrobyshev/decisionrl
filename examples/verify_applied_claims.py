"""Multi-seed verification of every applied-RL claim in the README.

Trains the appropriate agent over several seeds on each applied task and compares
the learned return against the *strong* classical baseline (exhaustive base-stock,
best value threshold, greedy price-threshold battery, best fixed price) from
:mod:`decisionrl.baselines`. Reports the interquartile mean (IQM) with a 95% bootstrap
confidence interval (rliable-style) rather than a single-seed number, and writes the
full record to JSON. This is what makes the "RL wins / RL matches" claims defensible.

Each (task, seed) is trained in its own short-lived subprocess and checkpointed to the
output JSON the moment it finishes. A re-run resumes from the last completed seed, and a
crashed seed is retried automatically. Isolating every run in a fresh process keeps a
long CPU sweep robust: no single failure can discard work already done, and the driver
never accumulates the native state that makes long in-process PyTorch runs flaky on some
platforms.

Run:            python examples/verify_applied_claims.py [--seeds 5] [--check]
(internal)      python examples/verify_applied_claims.py --worker --task KEY --seed N
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

OUT_PATH = Path("verify_applied_claims.json")
RETRIES = 4


def _build():
    """Import the library lazily so the driver stays importable without torch loaded."""
    from decisionrl import baselines as B
    from decisionrl.algorithms import DQN, PPO, SAC
    from decisionrl.envs import (
        DynamicPricing,
        EnergyMicrogrid,
        InventoryManagement,
        NonstationaryInventory,
        QueueAdmissionControl,
        SupplyChain,
        Thermostat,
    )

    # key: (group, label, agent, env, steps, kwargs, baseline label, baseline callable)
    return {
        # RL wins: the classical method breaks or is only a naive default.
        "nonstat": ("win", "Non-stationary inventory", DQN, NonstationaryInventory, 100_000,
                    dict(learning_rate=5e-4, buffer_size=50_000, learning_starts=1000,
                         target_update_interval=500),
                    "best fixed base-stock",
                    lambda: B.best_base_stock(NonstationaryInventory, seed=100)[1]),
        "supply": ("win", "Supply chain (2-echelon)", SAC, SupplyChain, 20_000,
                   dict(learning_starts=1000, batch_size=256),
                   "per-echelon base-stock",
                   lambda: B.best_supply_base_stock(SupplyChain, seed=100)[1]),
        "queue": ("win", "Queue admission control", PPO, QueueAdmissionControl, 30_000,
                  dict(n_steps=512, batch_size=64, n_epochs=10),
                  "best value threshold",
                  lambda: B.best_value_threshold(QueueAdmissionControl, seed=100)[1]),
        "energy": ("win", "Energy microgrid (battery)", SAC, EnergyMicrogrid, 20_000,
                   dict(learning_starts=1000, batch_size=256),
                   "greedy price-threshold",
                   lambda: B.best_price_threshold_battery(EnergyMicrogrid, seed=100)[1]),
        "thermostat": ("win", "Thermostat / HVAC", SAC, Thermostat, 15_000,
                       dict(learning_starts=1000, batch_size=256),
                       "bang-bang",
                       lambda: B.rollout_return(Thermostat, B.bang_bang(), seed=100)),
        # RL matches: the classic tool is already optimal.
        "inventory": ("match", "Inventory (stationary)", PPO, InventoryManagement, 40_000,
                      dict(n_steps=1024, batch_size=64, n_epochs=10, ent_coef=0.01),
                      "base-stock (optimal)",
                      lambda: B.best_base_stock(InventoryManagement, seed=100)[1]),
        "pricing": ("match", "Dynamic pricing", PPO, DynamicPricing, 100_000,
                    dict(n_steps=1024, batch_size=64, n_epochs=10, ent_coef=0.005),
                    "best fixed price",
                    lambda: B.best_fixed_action(DynamicPricing, seed=100)[1]),
    }


ORDER = ["nonstat", "supply", "queue", "energy", "thermostat", "inventory", "pricing"]


def run_worker(key: str, seed: int, device: str) -> None:
    """Train one seed of one task and print the evaluation return as ``RESULT <float>``."""
    import torch

    from decisionrl.training import evaluate_policy
    from decisionrl.utils import Logger, set_seed

    torch.set_num_threads(1)
    _, _, agent_cls, env_fn, steps, kw, _, _ = _build()[key]
    set_seed(seed)
    agent = agent_cls(env_fn(), seed=seed, logger=Logger(verbose=0), device=device, **kw)
    agent.learn(steps)
    score = evaluate_policy(agent, env_fn(), n_episodes=20, seed=100)[0]
    print(f"RESULT {score:.6f}", flush=True)


def train_one(key: str, seed: int, device: str) -> float:
    """Spawn a fresh subprocess for a single (task, seed); retry on a native crash."""
    cmd = [sys.executable, __file__, "--worker", "--task", key, "--seed", str(seed),
           "--device", device]
    for attempt in range(1, RETRIES + 1):
        proc = subprocess.run(cmd, capture_output=True, text=True)
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT "):
                return float(line.split()[1])
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:] or ["(no output)"]
        print(f"  [{key} seed {seed}] attempt {attempt}/{RETRIES} failed "
              f"(exit {proc.returncode}): {tail[0]}", flush=True)
        time.sleep(2)  # let the OS settle before the next attempt
    raise RuntimeError(f"{key} seed {seed} failed after {RETRIES} attempts")


def load_checkpoint(n_seeds: int) -> dict:
    if OUT_PATH.exists():
        prev = json.loads(OUT_PATH.read_text())
        if prev.get("seeds") == n_seeds:
            prev.setdefault("seed_values", {})
            return prev
    return {"seeds": n_seeds, "seed_values": {}, "rows": []}


def driver(n_seeds: int, check: bool, device: str) -> None:
    from decisionrl.evaluation import bootstrap_ci, iqm

    specs = _build()
    seeds = list(range(n_seeds))
    t0 = time.time()
    out = load_checkpoint(n_seeds)
    sv = out["seed_values"]
    print(f"training on device: {device}", flush=True)

    for key in ORDER:
        group, label, _, _, _, _, bname, baseline_fn = specs[key]
        got = sv.setdefault(key, {})
        for s in seeds:
            if str(s) in got:
                continue
            got[str(s)] = train_one(key, s, device)
            OUT_PATH.write_text(json.dumps(out, indent=2))  # checkpoint per seed
            print(f"  {label} seed {s}: {got[str(s)]:.1f}", flush=True)
        vals = [got[str(s)] for s in seeds]
        point, lo, hi = bootstrap_ci(vals, aggregate=iqm, reps=5000, seed=0)
        out["rows"] = [r for r in out["rows"] if r["task"] != label]
        out["rows"].append({"group": group, "task": label, "rl_iqm": point,
                            "ci_low": lo, "ci_high": hi, "rl_mean": statistics.mean(vals),
                            "baseline": bname, "baseline_value": baseline_fn(),
                            "seed_values": vals})
        OUT_PATH.write_text(json.dumps(out, indent=2))

    out["rows"].sort(key=lambda r: [specs[k][1] for k in ORDER].index(r["task"]))
    OUT_PATH.write_text(json.dumps(out, indent=2))

    hdr_task, hdr_rl, hdr_base = "task", "RL: IQM [95% CI]", "baseline"
    print(f"\n{hdr_task:<28}{hdr_rl:<26}{hdr_base:<24}verdict")
    print("-" * 92)
    for r in out["rows"]:
        point, lo, hi, bval = r["rl_iqm"], r["ci_low"], r["ci_high"], r["baseline_value"]
        verdict = ("WIN" if point > bval else "check") if r["group"] == "win" else "~match"
        rl_cell = f"{point:.1f} [{lo:.1f}, {hi:.1f}]"
        base_cell = f"{bval:.1f} ({r['baseline']})"
        print(f"{r['task']:<28}{rl_cell:<26}{base_cell:<24}{verdict}")
    print(f"\n{n_seeds} seeds each, IQM with 95% bootstrap CI, "
          f"reproduced on CPU in {time.time() - t0:.0f}s.")

    if check:
        regressed = [r["task"] for r in out["rows"]
                     if r["group"] == "win" and r["rl_iqm"] <= r["baseline_value"]]
        if regressed:
            print(f"\nREGRESSION: RL no longer beats the baseline on: {', '.join(regressed)}")
            raise SystemExit(1)
        print("\nAll 'RL wins' claims still hold.")


def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if any 'RL wins' claim regressed below its baseline")
    ap.add_argument("--device", default=None,
                    help="torch device for training (default: cuda if available, else cpu)")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--task", help=argparse.SUPPRESS)
    ap.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    args = ap.parse_args()
    device = args.device or _default_device()
    if args.worker:
        run_worker(args.task, args.seed, device)
    else:
        driver(args.seeds, args.check, device)


if __name__ == "__main__":
    main()
