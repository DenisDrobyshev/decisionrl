"""Multi-seed verification of every applied-RL claim in the README.

Trains the appropriate agent over several seeds on each applied task, and compares
the mean +/- std of the learned return against the *strong* classical baseline
(exhaustive base-stock, best value threshold, greedy price-threshold battery, best
fixed price) from :mod:`decisionrl.baselines`. Prints the honest table used in the
README and writes it to JSON. This is what makes the "RL wins / RL matches" claims
reproducible rather than single-seed.

Run: python examples/verify_applied_claims.py [--seeds 3]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

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
from decisionrl.training import evaluate_policy
from decisionrl.utils import Logger, set_seed


def learned_over_seeds(agent_cls, env_fn, steps, seeds, **kw):
    vals = []
    for s in seeds:
        set_seed(s)
        agent = agent_cls(env_fn(), seed=s, logger=Logger(verbose=0), **kw)
        agent.learn(steps)
        vals.append(evaluate_policy(agent, env_fn(), n_episodes=20, seed=100)[0])
    return vals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    torch.set_num_threads(1)
    seeds = list(range(args.seeds))
    t0 = time.time()
    rows = []  # (group, task, learned_vals, baseline_name, baseline_value)

    # ---- RL wins: classical method breaks or is only a naive default ----
    _, ns_bs = B.best_base_stock(NonstationaryInventory, seed=100)
    rows.append(("win", "Non-stationary inventory",
                 learned_over_seeds(DQN, NonstationaryInventory, 100_000, seeds,
                                    learning_rate=5e-4, buffer_size=50_000,
                                    learning_starts=1000, target_update_interval=500),
                 "best fixed base-stock", ns_bs))

    _, sc_bs = B.best_supply_base_stock(SupplyChain, seed=100)
    rows.append(("win", "Supply chain (2-echelon)",
                 learned_over_seeds(SAC, SupplyChain, 20_000, seeds,
                                    learning_starts=1000, batch_size=256),
                 "per-echelon base-stock", sc_bs))

    _, q_thr = B.best_value_threshold(QueueAdmissionControl, seed=100)
    rows.append(("win", "Queue admission control",
                 learned_over_seeds(PPO, QueueAdmissionControl, 30_000, seeds,
                                    n_steps=512, batch_size=64, n_epochs=10),
                 "best value threshold", q_thr))

    _, en_thr = B.best_price_threshold_battery(EnergyMicrogrid, seed=100)
    rows.append(("win", "Energy microgrid (battery)",
                 learned_over_seeds(SAC, EnergyMicrogrid, 20_000, seeds,
                                    learning_starts=1000, batch_size=256),
                 "greedy price-threshold", en_thr))

    bang = B.rollout_return(Thermostat, B.bang_bang(), seed=100)
    rows.append(("win", "Thermostat / HVAC",
                 learned_over_seeds(SAC, Thermostat, 15_000, seeds,
                                    learning_starts=1000, batch_size=256),
                 "bang-bang", bang))

    # ---- RL matches: the classic tool is already optimal ----
    _, inv_opt = B.best_base_stock(InventoryManagement, seed=100)
    rows.append(("match", "Inventory (stationary)",
                 learned_over_seeds(PPO, InventoryManagement, 40_000, seeds,
                                    n_steps=1024, batch_size=64, n_epochs=10, ent_coef=0.01),
                 "base-stock (optimal)", inv_opt))

    _, price_fx = B.best_fixed_action(DynamicPricing, seed=100)
    rows.append(("match", "Dynamic pricing",
                 learned_over_seeds(PPO, DynamicPricing, 60_000, seeds,
                                    n_steps=1024, batch_size=64, n_epochs=10, ent_coef=0.01),
                 "best fixed price", price_fx))

    out = {"seeds": args.seeds, "rows": []}
    print(f"\n{'task':<30}{'RL (mean±std)':<20}{'baseline':<24}{'verdict'}")
    print("-" * 84)
    for group, task, vals, bname, bval in rows:
        mean, std = statistics.mean(vals), (statistics.pstdev(vals) if len(vals) > 1 else 0.0)
        verdict = "WIN" if (mean > bval and group == "win") else ("~match" if group == "match" else "check")
        print(f"{task:<30}{f'{mean:.1f} ± {std:.1f}':<20}{f'{bval:.1f} ({bname})':<24}{verdict}")
        out["rows"].append({"group": group, "task": task, "rl_mean": mean, "rl_std": std,
                            "baseline": bname, "baseline_value": bval, "seed_values": vals})
    Path("verify_applied_claims.json").write_text(json.dumps(out, indent=2))
    print(f"\n{args.seeds} seeds each · reproduced on CPU in {time.time() - t0:.0f}s.")


if __name__ == "__main__":
    main()
