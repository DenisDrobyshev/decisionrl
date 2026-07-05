"""Train SAC (with automatic entropy tuning) on the built-in Pendulum.

Run: python examples/train_sac_pendulum.py
"""

from reinforce.algorithms import SAC
from reinforce.envs import Pendulum
from reinforce.training import evaluate_policy
from reinforce.utils import set_seed


def main() -> None:
    set_seed(0)
    agent = SAC(Pendulum(), learning_starts=1000, batch_size=256, seed=0)
    agent.learn(total_steps=20_000)

    mean, std = evaluate_policy(agent, Pendulum(), n_episodes=10)
    print(f"\nFinal evaluation return: {mean:.1f} +/- {std:.1f} (closer to 0 is better)")


if __name__ == "__main__":
    main()
