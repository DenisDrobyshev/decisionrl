"""Train agents on the complex scenarios and record animated GIFs for the README.

Writes to docs/assets/:
    scenario_reacher.gif      ReacherArm (SAC)
    scenario_navigation.gif   Navigation2D maze with lidar (SAC)
    scenario_lander.gif       LunarLander (PPO)

Uses the GPU automatically. Run: python examples/record_scenario_gifs.py
"""

from __future__ import annotations

import os

import torch

from decisionrl.algorithms import PPO, SAC
from decisionrl.envs import LunarLander, Navigation2D, ReacherArm
from decisionrl.utils import Logger, record_gif, set_seed

ASSETS = os.path.join(os.path.dirname(__file__), "..", "docs", "assets")
os.makedirs(ASSETS, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _gif(name, env_cls, agent, steps, fps=25):
    agent.learn(steps)
    path = os.path.join(ASSETS, name)
    record_gif(agent, env_cls(), path, max_steps=250, fps=fps, seed=7)
    print("wrote", name)


def main() -> None:
    set_seed(0)
    _gif("scenario_reacher.gif", ReacherArm,
         SAC(ReacherArm(), device=DEVICE, seed=0, logger=Logger(verbose=0)), 30_000)
    set_seed(0)
    _gif("scenario_navigation.gif", Navigation2D,
         SAC(Navigation2D(), device=DEVICE, seed=0, logger=Logger(verbose=0)), 60_000)
    set_seed(0)
    _gif("scenario_lander.gif", LunarLander,
         PPO(LunarLander(), n_steps=1024, batch_size=128, device=DEVICE, seed=0, logger=Logger(verbose=0)),
         200_000)


if __name__ == "__main__":
    main()
