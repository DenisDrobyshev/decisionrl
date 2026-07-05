"""Batteries-included environments (no external dependencies required).

Also exposes optional Gymnasium interop via :func:`make_gym` / :class:`GymAdapter`.
"""

from .bandit import MultiArmedBandit
from .cartpole import CartPole
from .grid_world import GridWorld
from .gym import GymAdapter, convert_space, make_gym
from .pendulum import Pendulum
from .point_mass import PointMass

__all__ = [
    "GridWorld",
    "MultiArmedBandit",
    "CartPole",
    "Pendulum",
    "PointMass",
    "GymAdapter",
    "make_gym",
    "convert_space",
]
