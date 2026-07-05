"""Train PPO on a vectorized Gymnasium environment.

Requires the optional extra: pip install "reinforce[gym]"
Run: python examples/ppo_gymnasium_vectorized.py
"""

from reinforce.algorithms import PPO
from reinforce.envs import make_gym
from reinforce.training import evaluate_policy
from reinforce.utils import set_seed
from reinforce.wrappers import SyncVectorEnv


def main() -> None:
    set_seed(0)
    venv = SyncVectorEnv([lambda: make_gym("CartPole-v1") for _ in range(8)])
    agent = PPO(venv, n_steps=256, batch_size=64, n_epochs=10, seed=0)
    agent.learn(total_steps=100_000)

    mean, std = evaluate_policy(agent, make_gym("CartPole-v1"), n_episodes=20)
    print(f"\nFinal evaluation return: {mean:.1f} +/- {std:.1f}")


if __name__ == "__main__":
    main()
