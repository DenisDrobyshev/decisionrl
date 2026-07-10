"""Train agents on the complex, varied scenario environments.

Four distinct domains, each solved with an appropriate algorithm:

    ReacherArm          (robotic manipulation)  -> SAC
    Navigation2D        (maze navigation)       -> SAC
    LunarLander         (rocket landing)        -> PPO
    PortfolioAllocation (finance)               -> SAC   (vs equal-weight baseline)

Uses the GPU automatically when a CUDA build of PyTorch is installed.

Run: python examples/complex_scenarios.py
"""

from __future__ import annotations

import numpy as np
import torch

from reinforce.algorithms import PPO, SAC
from reinforce.envs import LunarLander, Navigation2D, PortfolioAllocation, ReacherArm
from reinforce.training import evaluate_policy
from reinforce.utils import Logger, set_seed

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _report(name, before, after, baseline=None):
    line = f"{name:20s}  before={before:8.2f}  ->  after={after:8.2f}"
    if baseline is not None:
        line += f"   (baseline={baseline:8.2f})"
    print(line)


def reacher():
    set_seed(0)
    agent = SAC(ReacherArm(), device=DEVICE, seed=0, logger=Logger(verbose=0))
    before, _ = evaluate_policy(agent, ReacherArm(), n_episodes=20, seed=100)
    agent.learn(60_000)
    after, _ = evaluate_policy(agent, ReacherArm(), n_episodes=20, seed=100)
    _report("ReacherArm (SAC)", before, after)


def navigation():
    set_seed(0)
    agent = SAC(Navigation2D(), device=DEVICE, seed=0, logger=Logger(verbose=0))
    before, _ = evaluate_policy(agent, Navigation2D(), n_episodes=20, seed=100)
    agent.learn(80_000)
    after, _ = evaluate_policy(agent, Navigation2D(), n_episodes=20, seed=100)
    _report("Navigation2D (SAC)", before, after)


def lander():
    set_seed(0)
    agent = PPO(LunarLander(), n_steps=1024, batch_size=128, device=DEVICE, seed=0, logger=Logger(verbose=0))
    before, _ = evaluate_policy(agent, LunarLander(), n_episodes=20, seed=100)
    agent.learn(300_000)
    after, _ = evaluate_policy(agent, LunarLander(), n_episodes=20, seed=100)
    _report("LunarLander (PPO)", before, after)


def portfolio():
    set_seed(0)

    def make():
        return PortfolioAllocation(momentum=0.7, vol=0.03)

    agent = SAC(make(), device=DEVICE, seed=0, logger=Logger(verbose=0))
    before, _ = evaluate_policy(agent, make(), n_episodes=20, seed=100)
    agent.learn(60_000)
    after, _ = evaluate_policy(agent, make(), n_episodes=20, seed=100)

    # equal-weight (rebalance-to-uniform) baseline
    eq_returns = []
    env = make()
    for ep in range(20):
        obs, _ = env.reset(seed=100 + ep)
        done, tot = False, 0.0
        while not done:
            obs, r, term, trunc, _ = env.step(np.zeros(env.n, dtype=np.float32))  # softmax(0)=uniform
            tot += r
            done = term or trunc
        eq_returns.append(tot)
    _report("PortfolioAllocation (SAC)", before, after, baseline=float(np.mean(eq_returns)))


if __name__ == "__main__":
    print(f"device: {DEVICE}\n")
    reacher()
    navigation()
    lander()
    portfolio()
