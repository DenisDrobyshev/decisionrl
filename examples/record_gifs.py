"""Train agents and record episode GIFs for the README (docs/assets/*.gif).

Run: python examples/record_gifs.py
"""

from __future__ import annotations

import os

from decisionrl.algorithms import PPO, SAC, QLearning
from decisionrl.envs import CartPole, GridWorld, Pendulum
from decisionrl.training import evaluate_policy
from decisionrl.utils import record_gif, set_seed

ASSETS = os.path.join(os.path.dirname(__file__), "..", "docs", "assets")
os.makedirs(ASSETS, exist_ok=True)


def main() -> None:
    # CartPole balanced by PPO
    set_seed(0)
    ppo = PPO(CartPole(), n_steps=1024, batch_size=64, n_epochs=10, seed=0)
    ppo.learn(40_000)
    print("PPO CartPole:", evaluate_policy(ppo, CartPole(), n_episodes=10))
    record_gif(ppo, CartPole(max_steps=200), os.path.join(ASSETS, "cartpole_ppo.gif"),
               max_steps=200, fps=30, seed=1)

    # Pendulum swing-up by SAC
    set_seed(0)
    sac = SAC(Pendulum(), learning_starts=1000, batch_size=256, seed=0)
    sac.learn(20_000)
    print("SAC Pendulum:", evaluate_policy(sac, Pendulum(), n_episodes=10))
    record_gif(sac, Pendulum(max_steps=200), os.path.join(ASSETS, "pendulum_sac.gif"),
               max_steps=200, fps=30, seed=1)

    # GridWorld navigation by tabular Q-Learning
    set_seed(0)
    q = QLearning(GridWorld(rows=5, cols=5, start=(0, 0), goal=(4, 4)), learning_rate=0.2, seed=0)
    q.learn(40_000)
    record_gif(q, GridWorld(rows=5, cols=5, start=(0, 0), goal=(4, 4), max_steps=30),
               os.path.join(ASSETS, "gridworld_qlearning.gif"), max_steps=30, fps=4, seed=1)

    print("Saved GIFs to", os.path.abspath(ASSETS))


if __name__ == "__main__":
    main()
