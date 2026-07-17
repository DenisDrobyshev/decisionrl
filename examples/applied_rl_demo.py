"""Single-seed proof table for the applied environments: RL vs the strong baseline.

A quick (one-seed) version of the honest comparison. For the multi-seed mean ± std
numbers shown in the README, run ``examples/verify_applied_claims.py`` instead.

Two groups:
* **RL wins** — the classical method breaks (non-stationarity, coupled decisions):
  non-stationary inventory, supply chain, energy arbitrage, queue admission, HVAC.
* **RL matches** — the classic tool is already optimal: stationary inventory
  (base-stock) and dynamic pricing (best fixed price).

Baselines are the *strong* ones from ``decisionrl.baselines`` (best base-stock, best
value threshold, greedy price-threshold battery, best fixed price), not straw men.

Run: python examples/applied_rl_demo.py
"""

from __future__ import annotations

import time

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


def train(agent_cls, env_fn, steps, **kw) -> float:
    set_seed(0)
    agent = agent_cls(env_fn(), seed=0, logger=Logger(verbose=0), **kw)
    agent.learn(steps)
    return evaluate_policy(agent, env_fn(), n_episodes=40, seed=1)[0]


def main() -> None:
    torch.set_num_threads(1)
    t0 = time.time()
    wins, matches = [], []

    # ---- RL wins (classical method breaks) ----
    wins.append(("Non-stationary inventory",
                 train(DQN, NonstationaryInventory, 100_000, learning_rate=5e-4,
                       buffer_size=50_000, learning_starts=1000, target_update_interval=500),
                 "best fixed base-stock", B.best_base_stock(NonstationaryInventory)[1]))
    wins.append(("Energy microgrid (battery)",
                 train(SAC, EnergyMicrogrid, 20_000, learning_starts=1000, batch_size=256),
                 "greedy price-threshold", B.best_price_threshold_battery(EnergyMicrogrid)[1]))
    wins.append(("Supply chain (2-echelon)",
                 train(SAC, SupplyChain, 20_000, learning_starts=1000, batch_size=256),
                 "per-echelon base-stock", B.best_supply_base_stock(SupplyChain)[1]))
    wins.append(("Queue admission control",
                 train(PPO, QueueAdmissionControl, 30_000, n_steps=512, batch_size=64, n_epochs=10),
                 "best value threshold", B.best_value_threshold(QueueAdmissionControl)[1]))
    wins.append(("Thermostat / HVAC",
                 train(SAC, Thermostat, 15_000, learning_starts=1000, batch_size=256),
                 "bang-bang", B.rollout_return(Thermostat, B.bang_bang())))

    # ---- RL matches the known optimum ----
    matches.append(("Inventory (stationary)",
                    train(PPO, InventoryManagement, 40_000, n_steps=1024, batch_size=64,
                          n_epochs=10, ent_coef=0.01),
                    "base-stock (optimal)", B.best_base_stock(InventoryManagement)[1]))
    matches.append(("Dynamic pricing",
                    train(PPO, DynamicPricing, 60_000, n_steps=1024, batch_size=64,
                          n_epochs=10, ent_coef=0.01),
                    "best fixed price", B.best_fixed_action(DynamicPricing)[1]))

    def show(title, rows):
        print(f"\n### {title}")
        print("| Applied task | Learned (RL) | Baseline |")
        print("|---|---:|---:|")
        for task, learned, bname, bval in rows:
            print(f"| {task} | **{learned:.1f}** | {bval:.1f} · {bname} |")

    show("RL wins (classical method breaks)", wins)
    show("RL matches the known optimum (sanity)", matches)
    print(f"\nSingle seed · reproduced on CPU in {time.time() - t0:.0f}s. "
          f"For mean ± std over seeds, see examples/verify_applied_claims.py.")


if __name__ == "__main__":
    main()
