"""Train DQN (Double + Dueling) on a one-hot GridWorld.

Run: python examples/train_dqn_gridworld.py
"""

from reinforce.algorithms import DQN
from reinforce.envs import GridWorld
from reinforce.training import evaluate_policy
from reinforce.utils import set_seed


def make_env():
    return GridWorld(rows=5, cols=5, one_hot=True)


def main() -> None:
    set_seed(0)
    agent = DQN(
        make_env(),
        learning_rate=1e-3,
        learning_starts=500,
        target_update_interval=200,
        hidden_sizes=(64, 64),
        double_q=True,
        dueling=True,
        seed=0,
    )
    agent.learn(total_steps=20_000)

    mean, std = evaluate_policy(agent, make_env(), n_episodes=20)
    print(f"\nFinal evaluation return: {mean:.3f} +/- {std:.3f}")


if __name__ == "__main__":
    main()
