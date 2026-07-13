"""Decision Transformer: offline RL as return-conditioned sequence modeling.

Builds a mixed-quality offline dataset of CartPole trajectories (random ... expert),
trains a Decision Transformer by supervised sequence modeling, then shows that the
achieved return tracks the *target return* you condition on.

Run: python examples/decision_transformer.py
"""

from __future__ import annotations

import numpy as np

from decisionrl import collect_trajectories
from decisionrl.algorithms import DecisionTransformer
from decisionrl.data import TrajectoryDataset
from decisionrl.envs import CartPole
from decisionrl.utils import set_seed


def heuristic(o):
    return 1 if (o[2] + 0.5 * o[3]) > 0 else 0


def main() -> None:
    set_seed(0)
    rng = np.random.default_rng(0)
    env = CartPole()

    def make_policy(eps):
        return lambda o: env.action_space.sample() if rng.random() < eps else heuristic(o)

    print("collecting mixed-quality offline trajectories ...")
    trajs = []
    for eps in [0.0, 0.1, 0.3, 0.6, 1.0]:
        trajs += collect_trajectories(env, make_policy(eps), 30, seed=int(eps * 1000) + 1).trajectories
    data = TrajectoryDataset(trajs, discrete=True, seed=0)
    print(f"dataset: {len(data)} trajectories, returns {data.returns.min():.0f}..{data.returns.max():.0f}")

    dt = DecisionTransformer(env, context_len=20, embed_dim=128, n_layers=3, max_ep_len=500, seed=0)
    print("training Decision Transformer ...")
    dt.learn_offline(data, n_iters=3000, batch_size=64)

    print("\nreturn-conditioned evaluation:")
    for target in [500, 400, 250, 100, 50]:
        mean, std = dt.evaluate(CartPole(), target_return=target, n_episodes=15, seed=100)
        print(f"  target={target:4d}  ->  achieved {mean:6.1f} +/- {std:5.1f}")


if __name__ == "__main__":
    main()
