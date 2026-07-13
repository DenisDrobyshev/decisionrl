"""Intrinsic motivation: exploration bonuses that work with any agent.

Wraps an environment with a curiosity module (RND or ICM) so the agent receives an
intrinsic novelty bonus on top of the extrinsic reward. Demonstrates that RND's
novelty signal is high for unseen states and decays as states become familiar —
the mechanism that drives exploration on sparse-reward tasks.

Run: python examples/curiosity_exploration.py
"""

from __future__ import annotations

import numpy as np

from decisionrl.algorithms import DQN
from decisionrl.envs import CartPole
from decisionrl.exploration import ICM, RND, CuriosityWrapper
from decisionrl.utils import Logger, set_seed


def rnd_novelty_demo() -> None:
    set_seed(0)
    rnd = RND(obs_dim=4)
    seen = np.ones(4, dtype=np.float32)
    unseen = np.full(4, -3.0, dtype=np.float32)
    print(f"novelty(new state)               = {rnd.intrinsic_reward(None, None, seen):.4f}")
    for _ in range(300):
        rnd.update(None, None, seen)
    print(f"novelty(same state, seen 300x)   = {rnd.intrinsic_reward(None, None, seen):.5f}")
    print(f"novelty(a different, unseen state) = {rnd.intrinsic_reward(None, None, unseen):.4f}")


def train_with_curiosity() -> None:
    set_seed(0)
    for name, module in [("RND", RND(obs_dim=4)), ("ICM", ICM(obs_dim=4, action_space=CartPole().action_space))]:
        env = CuriosityWrapper(CartPole(), module, intrinsic_coef=0.5)
        agent = DQN(env, learning_starts=100, seed=0, logger=Logger(verbose=0))
        agent.learn(5_000)
        print(f"[{name}] trained DQN on curiosity-wrapped CartPole for {agent.num_timesteps} steps")


if __name__ == "__main__":
    rnd_novelty_demo()
    train_with_curiosity()
