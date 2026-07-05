"""Applied RL demos: inventory management (PPO) and HVAC/thermostat (SAC).

Trains an agent on each *applied* task, simulates a full episode of the learned
policy against a naive baseline, and renders the figures used in the README:

    docs/assets/applied_inventory.png
    docs/assets/applied_thermostat.png

Run: python examples/applied_demo.py
"""

from __future__ import annotations

import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from reinforce.algorithms import PPO, SAC
from reinforce.envs import InventoryManagement, Thermostat
from reinforce.training import EvalCallback, evaluate_policy
from reinforce.utils import Logger, set_seed

ASSETS = os.path.join(os.path.dirname(__file__), "..", "docs", "assets")
os.makedirs(ASSETS, exist_ok=True)
C = {"agent": "#2563eb", "agent2": "#db2777", "baseline": "#94a3b8", "accent": "#f59e0b", "ok": "#16a34a"}


def episode_return(env_fn, policy, episodes=30, seed=1):
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


# --------------------------------------------------------------------------- #
# 1) Inventory management with PPO
# --------------------------------------------------------------------------- #
def inventory_demo(ax_sim, ax_curve):
    set_seed(0)
    eval_freq = 2000
    eval_cb = EvalCallback(InventoryManagement(), eval_freq=eval_freq, n_eval_episodes=10, verbose=0)
    agent = PPO(InventoryManagement(), n_steps=1024, batch_size=64, n_epochs=10,
                ent_coef=0.01, seed=0, logger=Logger(verbose=0))
    agent.learn(40_000, callback=eval_cb)

    learned = evaluate_policy(agent, InventoryManagement(), n_episodes=30, seed=1)[0]
    rnd = episode_return(InventoryManagement, lambda e, o: e.action_space.sample())

    def basestock(S):
        return lambda e, o: int(np.clip(S - round(o[0] * e.max_inventory), 0, e.max_order))

    best_bs = max((episode_return(InventoryManagement, basestock(s)) for s in range(4, 12)))

    # simulate one episode of the learned policy
    env = InventoryManagement()
    obs, _ = env.reset(seed=123)
    inv, orders, demand, done = [], [], [], False
    while not done:
        a = agent.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(a)
        inv.append(env._inventory)
        orders.append(info["order"])
        demand.append(info["demand"])
        done = term or trunc
    weeks = np.arange(len(inv))

    ax_sim.bar(weeks, orders, color=C["accent"], alpha=0.6, label="units ordered")
    ax_sim.plot(weeks, inv, color=C["agent"], lw=2, marker="o", ms=3, label="inventory level")
    ax_sim.plot(weeks, demand, color=C["baseline"], lw=1.2, ls="--", label="demand")
    ax_sim.set_title("PPO inventory policy - one simulated episode", fontsize=11, fontweight="bold")
    ax_sim.set_xlabel("week")
    ax_sim.set_ylabel("units")
    ax_sim.legend(fontsize=8, loc="upper right")

    xs = [eval_freq * (i + 1) for i in range(len(eval_cb.evaluations))]
    ax_curve.plot(xs, eval_cb.evaluations, color=C["agent"], lw=2, marker="o", ms=3, label="PPO (evaluated)")
    ax_curve.axhline(best_bs, color=C["ok"], ls="--", lw=1.5, label=f"best base-stock heuristic ({best_bs:.0f})")
    ax_curve.axhline(rnd, color=C["baseline"], ls=":", lw=1.5, label=f"random policy ({rnd:.0f})")
    ax_curve.set_title("Learning curve - profit per episode", fontsize=11, fontweight="bold")
    ax_curve.set_xlabel("environment steps")
    ax_curve.set_ylabel("profit")
    ax_curve.legend(fontsize=8, loc="lower right")
    return learned, rnd, best_bs


# --------------------------------------------------------------------------- #
# 2) Thermostat / HVAC with SAC
# --------------------------------------------------------------------------- #
def thermostat_demo(ax_sim, ax_curve):
    set_seed(0)
    eval_freq = 1000
    eval_cb = EvalCallback(Thermostat(), eval_freq=eval_freq, n_eval_episodes=5, verbose=0)
    agent = SAC(Thermostat(), learning_starts=1000, batch_size=256, seed=0, logger=Logger(verbose=0))
    agent.learn(15_000, callback=eval_cb)

    learned = evaluate_policy(agent, Thermostat(), n_episodes=20, seed=1)[0]
    bang = episode_return(Thermostat, lambda e, o: np.array([1.0 if o[0] < 0 else -1.0], np.float32))

    def rollout(policy):
        env = Thermostat()
        obs, _ = env.reset(seed=7)
        indoor, outdoor, power, done = [], [], [], False
        while not done:
            obs, _, term, trunc, info = env.step(policy(obs))
            indoor.append(info["indoor"])
            outdoor.append(info["outdoor"])
            power.append(info["power"])
            done = term or trunc
        return np.array(indoor), np.array(outdoor), np.array(power)

    sac_in, outdoor, sac_pw = rollout(lambda o: agent.predict(o, deterministic=True))
    bang_in, _, bang_pw = rollout(lambda o: np.array([1.0 if o[0] < 0 else -1.0], np.float32))
    t = np.arange(len(sac_in))
    setpoint = Thermostat().setpoint

    ax_sim.axhspan(setpoint - 1, setpoint + 1, color=C["ok"], alpha=0.12, label="comfort band")
    ax_sim.axhline(setpoint, color=C["ok"], lw=1, ls="--")
    ax_sim.plot(t, outdoor, color=C["baseline"], lw=1.2, label="outdoor temp")
    ax_sim.plot(t, bang_in, color=C["accent"], lw=1.2, alpha=0.8, label=f"bang-bang (E={np.sum(bang_pw**2):.0f})")
    ax_sim.plot(t, sac_in, color=C["agent2"], lw=2, label=f"SAC (E={np.sum(sac_pw**2):.0f})")
    ax_sim.set_title("SAC thermostat - indoor temperature tracking", fontsize=11, fontweight="bold")
    ax_sim.set_xlabel("time step")
    ax_sim.set_ylabel("temperature (deg C)")
    ax_sim.legend(fontsize=8, loc="upper right")

    xs = [eval_freq * (i + 1) for i in range(len(eval_cb.evaluations))]
    ax_curve.plot(xs, eval_cb.evaluations, color=C["agent2"], lw=2, marker="o", ms=3, label="SAC (evaluated)")
    ax_curve.axhline(bang, color=C["accent"], ls="--", lw=1.5, label=f"bang-bang baseline ({bang:.0f})")
    # Warmup evaluations reach ~-40000; clip so the region where SAC overtakes
    # the baseline is legible.
    ax_curve.set_ylim(-1500, 100)
    ax_curve.set_title("Learning curve - evaluated return (y-axis clipped)", fontsize=11, fontweight="bold")
    ax_curve.set_xlabel("environment steps")
    ax_curve.set_ylabel("return (higher = better)")
    ax_curve.legend(fontsize=8, loc="lower right")
    return learned, bang


def main() -> None:
    t0 = time.time()
    plt.style.use("seaborn-v0_8-whitegrid")

    print("Training PPO on InventoryManagement ...")
    fig1, (a1, a2) = plt.subplots(2, 1, figsize=(9, 7))
    fig1.suptitle("Applied RL #1 - Inventory management", fontsize=14, fontweight="bold")
    inv_learned, inv_rnd, inv_bs = inventory_demo(a1, a2)
    fig1.tight_layout(rect=(0, 0, 1, 0.96))
    p1 = os.path.join(ASSETS, "applied_inventory.png")
    fig1.savefig(p1, dpi=130, bbox_inches="tight")
    plt.close(fig1)

    print("Training SAC on Thermostat ...")
    fig2, (b1, b2) = plt.subplots(2, 1, figsize=(9, 7))
    fig2.suptitle("Applied RL #2 - Thermostat / HVAC control", fontsize=14, fontweight="bold")
    th_learned, th_bang = thermostat_demo(b1, b2)
    fig2.tight_layout(rect=(0, 0, 1, 0.96))
    p2 = os.path.join(ASSETS, "applied_thermostat.png")
    fig2.savefig(p2, dpi=130, bbox_inches="tight")
    plt.close(fig2)

    print("\n" + "=" * 64)
    print(f"{'applied task':<28}{'learned':>10}{'baseline':>12}{'random':>12}")
    print("-" * 64)
    print(f"{'Inventory (PPO)':<28}{inv_learned:>10.1f}{inv_bs:>12.1f}{inv_rnd:>12.1f}")
    print(f"{'Thermostat (SAC)':<28}{th_learned:>10.1f}{th_bang:>12.1f}{'-':>12}")
    print("=" * 64)
    print(f"\nSaved {p1}\nSaved {p2}\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
