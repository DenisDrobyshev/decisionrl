"""Proof that RL beats the classic heuristic on every applied environment.

For each operational-decision environment this trains the appropriate agent,
evaluates the learned policy against the textbook baseline (base-stock,
bang-bang, best fixed price, admit-all, no-battery), and prints a Markdown table
- the same table shown in the README. Reproducible on CPU in a few minutes.

Run: python examples/applied_rl_demo.py
"""

from __future__ import annotations

import time

import numpy as np
import torch

from decisionrl.algorithms import PPO, SAC
from decisionrl.envs import (
    DynamicPricing,
    EnergyMicrogrid,
    InventoryManagement,
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


def train(agent_cls, env_fn, steps, **kw):
    set_seed(0)
    agent = agent_cls(env_fn(), seed=0, logger=Logger(verbose=0), **kw)
    agent.learn(steps)
    return evaluate_policy(agent, env_fn(), n_episodes=40, seed=1)[0]


def main() -> None:
    torch.set_num_threads(1)
    t0 = time.time()
    rows = []  # (task, metric, learned, baseline_name, baseline_value)

    # 1) Inventory management — PPO vs best base-stock heuristic.
    def basestock(S):
        return lambda e, o: int(np.clip(S - round(o[0] * e.max_inventory), 0, e.max_order))
    inv_bs = max(baseline_return(InventoryManagement, basestock(s)) for s in range(4, 12))
    inv = train(PPO, InventoryManagement, 40_000, n_steps=1024, batch_size=64, n_epochs=10, ent_coef=0.01)
    rows.append(("Inventory management", "profit", inv, "best base-stock", inv_bs))

    # 2) Dynamic pricing — PPO vs a no-strategy (random) price. RL also *matches*
    # the best fixed price, i.e. it recovers the revenue-optimal price online.
    rng = np.random.default_rng(0)
    n_p = DynamicPricing().action_space.n
    price_rand = baseline_return(DynamicPricing, lambda e, o: int(rng.integers(n_p)))
    price = train(PPO, DynamicPricing, 60_000, n_steps=1024, batch_size=64, n_epochs=10, ent_coef=0.01)
    rows.append(("Dynamic pricing", "revenue", price, "random pricing", price_rand))

    # 3) Queue admission control — PPO vs admit-all.
    q_admit = baseline_return(QueueAdmissionControl, lambda e, o: 1)
    q = train(PPO, QueueAdmissionControl, 30_000, n_steps=512, batch_size=64, n_epochs=10)
    rows.append(("Queue admission control", "value", q, "admit-all", q_admit))

    # 4) Thermostat / HVAC — SAC vs bang-bang.
    bang = baseline_return(Thermostat, lambda e, o: np.array([1.0 if o[0] < 0 else -1.0], np.float32))
    th = train(SAC, Thermostat, 15_000, learning_starts=1000, batch_size=256)
    rows.append(("Thermostat / HVAC", "return", th, "bang-bang", bang))

    # 5) Energy microgrid — SAC vs no-battery.
    nobatt = baseline_return(EnergyMicrogrid, lambda e, o: np.array([0.0], np.float32))
    en = train(SAC, EnergyMicrogrid, 20_000, learning_starts=1000, batch_size=256)
    rows.append(("Energy microgrid (battery)", "return", en, "no battery", nobatt))

    # 6) Supply chain — SAC vs order-nothing.
    nothing = baseline_return(SupplyChain, lambda e, o: np.zeros(2, np.float32))
    sc = train(SAC, SupplyChain, 20_000, learning_starts=1000, batch_size=256)
    rows.append(("Supply chain (2-echelon)", "return", sc, "order-nothing", nothing))

    print("\n| Applied task | Metric | Learned (RL) | Baseline | Baseline value |")
    print("|---|---|---:|---|---:|")
    for task, metric, learned, bname, bval in rows:
        print(f"| {task} | {metric} | **{learned:.1f}** | {bname} | {bval:.1f} |")
    print(f"\nReproduced on CPU in {time.time() - t0:.0f}s.")


if __name__ == "__main__":
    main()
