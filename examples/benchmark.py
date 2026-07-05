"""Working demo: train several agents on applied tasks, then render figures.

This script doubles as an end-to-end smoke test of the whole library and as the
generator of the plots shown in the README. It trains:

* PPO on CartPole            (discrete control)
* DQN (Double+Dueling) on GridWorld (discrete navigation, function approx.)
* SAC on Pendulum            (continuous control, swing-up)
* tabular Q-Learning on GridWorld  (exact control)

and writes two images to ``docs/assets/``:
    learning_curves.png   -- return vs. environment steps for each agent
    gridworld_policy.png  -- the greedy policy learned by Q-Learning

Run: python examples/benchmark.py
"""

from __future__ import annotations

import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from reinforce.algorithms import DQN, PPO, SAC, QLearning
from reinforce.envs import CartPole, GridWorld, Pendulum
from reinforce.training import evaluate_policy
from reinforce.utils import Logger, set_seed

ASSETS = os.path.join(os.path.dirname(__file__), "..", "docs", "assets")
os.makedirs(ASSETS, exist_ok=True)

PALETTE = {
    "ppo": "#2563eb",
    "dqn": "#16a34a",
    "sac": "#db2777",
    "q": "#f59e0b",
}


class HistoryLogger(Logger):
    """A silent logger that keeps every dumped scalar for later plotting."""

    def __init__(self) -> None:
        super().__init__(verbose=0)
        self.history: dict[str, list] = {}

    def dump(self, step: int) -> None:
        for k, v in self._values.items():
            self.history.setdefault(k, []).append((step, v))
        self._values.clear()

    def curve(self, key: str = "rollout/ep_return_mean"):
        pts = self.history.get(key, [])
        if not pts:
            return np.array([]), np.array([])
        xs, ys = zip(*pts)
        return np.array(xs), np.array(ys)


def train_ppo():
    set_seed(0)
    log = HistoryLogger()
    agent = PPO(CartPole(), n_steps=1024, batch_size=64, n_epochs=10, seed=0, logger=log)
    agent.learn(45_000)
    mean, std = evaluate_policy(agent, CartPole(), n_episodes=20)
    return log, mean, std


def train_dqn():
    set_seed(0)
    log = HistoryLogger()

    def make():
        return GridWorld(rows=5, cols=5, one_hot=True)

    agent = DQN(make(), learning_rate=1e-3, learning_starts=500, batch_size=64,
                buffer_size=10_000, hidden_sizes=(64, 64), target_update_interval=200,
                double_q=True, dueling=True, seed=0, logger=log)
    agent.learn(20_000, log_interval=5)
    mean, std = evaluate_policy(agent, make(), n_episodes=20)
    return log, mean, std


def train_sac():
    set_seed(0)
    log = HistoryLogger()
    agent = SAC(Pendulum(), learning_starts=1000, batch_size=256, seed=0, logger=log)
    agent.learn(15_000, log_interval=5)
    mean, std = evaluate_policy(agent, Pendulum(), n_episodes=10)
    return log, mean, std


def train_q():
    set_seed(0)
    log = HistoryLogger()
    env = GridWorld(rows=4, cols=4, start=(0, 0), goal=(3, 3))
    agent = QLearning(env, learning_rate=0.2, seed=0, logger=log)
    agent.learn(30_000, log_interval=20)
    return log, agent, env


def plot_learning_curves(results) -> str:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle("reinforce - agents learning applied tasks", fontsize=15, fontweight="bold")

    panels = [
        (axes[0, 0], "ppo", "PPO - CartPole (discrete control)", results["ppo"][0], "return (max 500)"),
        (axes[0, 1], "dqn", "DQN - GridWorld 5x5 (navigation)", results["dqn"][0], "return"),
        (axes[1, 0], "sac", "SAC - Pendulum (continuous swing-up)", results["sac"][0], "return (higher=better)"),
        (axes[1, 1], "q", "Q-Learning - GridWorld 4x4 (tabular)", results["q"][0], "return"),
    ]
    for ax, key, title, log, ylabel in panels:
        xs, ys = log.curve()
        ax.plot(xs, ys, color=PALETTE[key], linewidth=2)
        ax.fill_between(xs, ys, ys.min() if len(ys) else 0, color=PALETTE[key], alpha=0.12)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("environment steps")
        ax.set_ylabel(ylabel)
        ax.margins(x=0.01)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = os.path.join(ASSETS, "learning_curves.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_gridworld_policy(agent, env) -> str:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(5, 5))
    arrows = {0: (0, 0.3), 1: (0.3, 0), 2: (0, -0.3), 3: (-0.3, 0)}  # up,right,down,left

    for r in range(env.rows):
        for c in range(env.cols):
            x, y = c, env.rows - 1 - r
            if (r, c) == env.goal:
                ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color="#16a34a", alpha=0.35))
                ax.text(x, y, "GOAL", ha="center", va="center", fontweight="bold", color="#065f46")
                continue
            if (r, c) == env.start:
                ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color="#2563eb", alpha=0.15))
            a = agent.predict(r * env.cols + c)
            dx, dy = arrows[a]
            ax.arrow(x, y, dx, dy, head_width=0.15, head_length=0.12, fc="#1e293b", ec="#1e293b")

    ax.set_xlim(-0.5, env.cols - 0.5)
    ax.set_ylim(-0.5, env.rows - 0.5)
    ax.set_xticks(range(env.cols))
    ax.set_yticks(range(env.rows))
    ax.set_aspect("equal")
    ax.set_title("Greedy policy learned by Q-Learning", fontsize=12, fontweight="bold")
    path = os.path.join(ASSETS, "gridworld_policy.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    t0 = time.time()
    results = {}
    print("Training PPO on CartPole ...")
    results["ppo"] = train_ppo()
    print("Training DQN on GridWorld ...")
    results["dqn"] = train_dqn()
    print("Training SAC on Pendulum ...")
    results["sac"] = train_sac()
    print("Training Q-Learning on GridWorld ...")
    q_log, q_agent, q_env = train_q()
    results["q"] = (q_log, None, None)

    print("\n" + "=" * 56)
    print(f"{'agent / task':<34}{'final eval return':>20}")
    print("-" * 56)
    print(f"{'PPO / CartPole':<34}{results['ppo'][1]:>13.1f} +/- {results['ppo'][2]:.1f}")
    print(f"{'DQN / GridWorld 5x5':<34}{results['dqn'][1]:>13.2f} +/- {results['dqn'][2]:.2f}")
    print(f"{'SAC / Pendulum':<34}{results['sac'][1]:>13.1f} +/- {results['sac'][2]:.1f}")
    print("=" * 56)

    p1 = plot_learning_curves(results)
    p2 = plot_gridworld_policy(q_agent, q_env)
    print(f"\nSaved {p1}")
    print(f"Saved {p2}")
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
