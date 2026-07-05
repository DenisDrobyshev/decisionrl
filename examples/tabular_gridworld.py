"""Solve a GridWorld with tabular Q-Learning and print the greedy policy.

Run: python examples/tabular_gridworld.py
"""

from reinforce.algorithms import QLearning
from reinforce.envs import GridWorld
from reinforce.utils import set_seed

ARROWS = {0: "^", 1: ">", 2: "v", 3: "<"}


def main() -> None:
    set_seed(0)
    env = GridWorld(rows=4, cols=4, start=(0, 0), goal=(3, 3))
    agent = QLearning(env, learning_rate=0.2, seed=0)
    agent.learn(total_steps=30_000)

    print("\nLearned greedy policy (G = goal):")
    for r in range(env.rows):
        row = []
        for c in range(env.cols):
            if (r, c) == env.goal:
                row.append("G")
            else:
                row.append(ARROWS[agent.predict(r * env.cols + c)])
        print(" ".join(row))


if __name__ == "__main__":
    main()
