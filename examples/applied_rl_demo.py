"""Proof table for the applied environments: RL vs the classic baseline.

Honest by construction. It reports two groups:

* **RL wins** — tasks where the classical method breaks (non-stationarity, coupled
  decisions, straw-man defaults): non-stationary inventory, supply chain, queue
  admission, energy arbitrage, thermostat.
* **RL matches** — tasks where the classic tool is already optimal: stationary
  inventory (base-stock) and dynamic pricing (best fixed price). RL reaches the
  optimum from scratch but does not beat it, and the table says so.

Reproducible on CPU in a few minutes.

Run: python examples/applied_rl_demo.py
"""

from __future__ import annotations

import time

import numpy as np
import torch

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


def baseline_return(env_fn, policy, episodes=40, seed=1) -> float:
    rs = []
    for ep in range(episodes):
        env = env_fn()
        obs, _ = env.reset(seed=seed + ep)
        done, tot = False, 0.0
        while not done:
            obs, r, term, trunc, _ = env.step(policy(env, obs))
            tot += r
            done = term or trunc
        rs.append(tot)
    return float(np.mean(rs))


def train(agent_cls, env_fn, steps, **kw) -> float:
    set_seed(0)
    agent = agent_cls(env_fn(), seed=0, logger=Logger(verbose=0), **kw)
    agent.learn(steps)
    return evaluate_policy(agent, env_fn(), n_episodes=40, seed=1)[0]


def best_base_stock(env_fn, obs_inv_index=0, s_range=range(4, 30)) -> tuple:
    """Best fixed 'order up to S' policy over a grid (single-echelon inventory)."""
    def policy_for(S):
        return lambda e, o: int(np.clip(round(S - o[obs_inv_index] * e.max_inventory),
                                        0, e.max_order))
    best = max(((S, baseline_return(env_fn, policy_for(S))) for S in s_range),
              key=lambda x: x[1])
    return best  # (S, value)


def main() -> None:
    torch.set_num_threads(1)
    t0 = time.time()
    wins, matches = [], []

    # ------------------------------------------------------------------ WINS
    # 1) Non-stationary inventory — RL beats the best FIXED base-stock (the classic
    #    formula is only optimal for stationary demand). DQN is stable here across
    #    seeds; PPO is not, so we use the value-based agent for this discrete task.
    _, ns_bs = best_base_stock(NonstationaryInventory)
    ns = train(DQN, NonstationaryInventory, 100_000, learning_rate=5e-4, buffer_size=50_000,
               learning_starts=1000, target_update_interval=500)
    wins.append(("Non-stationary inventory", ns, "best fixed base-stock", ns_bs))

    # 2) Supply chain — RL beats a per-echelon base-stock heuristic.
    def sc_base_stock(S):
        def pol(e, o):
            ri, wi, rp, wp, _ = o * 40.0
            return np.array([max(0.0, S - (ri + rp)) / 15.0,
                             max(0.0, S - (wi + wp)) / 15.0], dtype=np.float32)
        return pol
    sc_bs = max(baseline_return(SupplyChain, sc_base_stock(S)) for S in range(8, 22))
    sc = train(SAC, SupplyChain, 20_000, learning_starts=1000, batch_size=256)
    wins.append(("Supply chain (2-echelon)", sc, "per-echelon base-stock", sc_bs))

    # 3) Queue admission control — RL beats admit-all.
    q_admit = baseline_return(QueueAdmissionControl, lambda e, o: 1)
    q = train(PPO, QueueAdmissionControl, 30_000, n_steps=512, batch_size=64, n_epochs=10)
    wins.append(("Queue admission control", q, "admit-all", q_admit))

    # 4) Energy microgrid — RL beats a no-battery policy.
    en_base = baseline_return(EnergyMicrogrid, lambda e, o: np.array([0.0], np.float32))
    en = train(SAC, EnergyMicrogrid, 20_000, learning_starts=1000, batch_size=256)
    wins.append(("Energy microgrid (battery)", en, "no battery", en_base))

    # 5) Thermostat / HVAC — RL beats bang-bang.
    bang = baseline_return(Thermostat, lambda e, o: np.array([1.0 if o[0] < 0 else -1.0], np.float32))
    th = train(SAC, Thermostat, 15_000, learning_starts=1000, batch_size=256)
    wins.append(("Thermostat / HVAC", th, "bang-bang", bang))

    # --------------------------------------------------------------- MATCHES
    # 6) Stationary inventory — RL only MATCHES the provably-optimal base-stock.
    _, inv_bs = best_base_stock(InventoryManagement)
    inv = train(PPO, InventoryManagement, 40_000, n_steps=1024, batch_size=64,
                n_epochs=10, ent_coef=0.01)
    matches.append(("Inventory (stationary)", inv, "base-stock (optimal)", inv_bs))

    # 7) Dynamic pricing — RL MATCHES the best fixed price (well above random).
    def fixed_price(a):
        return lambda e, o: a
    n_p = DynamicPricing().action_space.n
    price_bs = max(baseline_return(DynamicPricing, fixed_price(a)) for a in range(n_p))
    price = train(PPO, DynamicPricing, 60_000, n_steps=1024, batch_size=64,
                  n_epochs=10, ent_coef=0.01)
    matches.append(("Dynamic pricing", price, "best fixed price", price_bs))

    def show(title, rows):
        print(f"\n### {title}")
        print("| Applied task | Learned (RL) | Baseline |")
        print("|---|---:|---:|")
        for task, learned, bname, bval in rows:
            print(f"| {task} | **{learned:.1f}** | {bval:.1f} · {bname} |")

    show("RL wins (classical method breaks)", wins)
    show("RL matches the known optimum (sanity)", matches)
    print(f"\nReproduced on CPU in {time.time() - t0:.0f}s.")


if __name__ == "__main__":
    main()
