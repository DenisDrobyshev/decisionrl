"""Train PPO on a vectorized Gymnasium environment.

Requires the optional extra: pip install "decisionrl[gym]"
Run: python examples/ppo_gymnasium_vectorized.py
"""

from decisionrl.algorithms import PPO
from decisionrl.envs import make_gym
from decisionrl.training import evaluate_policy
from decisionrl.utils import set_seed
from decisionrl.wrappers import SyncVectorEnv


def main() -> None:
    set_seed(0)
    venv = SyncVectorEnv([lambda: make_gym("CartPole-v1") for _ in range(8)])
    agent = PPO(venv, n_steps=256, batch_size=64, n_epochs=10, seed=0)
    agent.learn(total_steps=100_000)

    mean, std = evaluate_policy(agent, make_gym("CartPole-v1"), n_episodes=20)
    print(f"\nFinal evaluation return: {mean:.1f} +/- {std:.1f}")


if __name__ == "__main__":
    main()
