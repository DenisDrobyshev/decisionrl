"""Offline RL: collect a dataset with a behaviour policy, then train TD3+BC on it.

No environment interaction happens during training - the agent learns purely from
the recorded transitions. Run: python examples/offline_td3bc.py
"""

import numpy as np

from decisionrl.algorithms import TD3BC
from decisionrl.data import collect_dataset
from decisionrl.envs import PointMass
from decisionrl.training import evaluate_policy
from decisionrl.utils import set_seed


def main() -> None:
    set_seed(0)
    rng = np.random.default_rng(0)

    # A noisy scripted behaviour policy that mostly moves toward the goal.
    def behavior(obs):
        return np.clip(-np.asarray(obs) * 3 + 0.3 * rng.standard_normal(2), -1, 1).astype(np.float32)

    print("Collecting offline dataset ...")
    dataset = collect_dataset(PointMass(), behavior, n_transitions=20_000, gamma=0.99, seed=0)

    print("Training TD3+BC offline ...")
    agent = TD3BC(PointMass(), alpha=2.5, batch_size=256, seed=0)
    agent.learn_offline(dataset, total_steps=10_000)

    mean, std = evaluate_policy(agent, PointMass(), n_episodes=20)
    print(f"\nOffline TD3+BC return: {mean:.2f} +/- {std:.2f} (random policy ~ -42)")


if __name__ == "__main__":
    main()
