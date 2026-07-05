"""Batteries-included environments (no external dependencies required).

Also exposes optional Gymnasium interop via :func:`make_gym` / :class:`GymAdapter`.
"""

from .acrobot import Acrobot
from .bandit import MultiArmedBandit
from .cartpole import CartPole
from .grid_world import GridWorld
from .gym import GymAdapter, convert_space, make_gym
from .inventory import InventoryManagement
from .mountain_car import MountainCar, MountainCarContinuous
from .pendulum import Pendulum
from .point_mass import PointMass
from .thermostat import Thermostat

__all__ = [
    # classic / toy
    "GridWorld",
    "MultiArmedBandit",
    "CartPole",
    "Pendulum",
    "PointMass",
    "MountainCar",
    "MountainCarContinuous",
    "Acrobot",
    # applied
    "InventoryManagement",
    "Thermostat",
    # gymnasium interop
    "GymAdapter",
    "make_gym",
    "convert_space",
]
