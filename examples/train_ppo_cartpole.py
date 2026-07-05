"""Train PPO on the built-in CartPole (no external dependencies needed).

Run: python examples/train_ppo_cartpole.py
"""

from reinforce.algorithms import PPO
from reinforce.envs import CartPole
from reinforce.training import evaluate_policy
from reinforce.utils import set_seed


def main() -> None:
    set_seed(0)
    agent = PPO(CartPole(), n_steps=1024, batch_size=64, n_epochs=10, seed=0)
    agent.learn(total_steps=50_000)

    mean, std = evaluate_policy(agent, CartPole(), n_episodes=20)
    print(f"\nFinal evaluation return: {mean:.1f} +/- {std:.1f} (max possible = 500)")

    agent.save("ppo_cartpole.pt")
    print("Saved to ppo_cartpole.pt")


if __name__ == "__main__":
    main()
