"""Batteries-included environments (no external dependencies required).

Also exposes optional Gymnasium interop via :func:`make_gym` / :class:`GymAdapter`.
"""

from .acrobot import Acrobot
from .bandit import BernoulliBandit, MultiArmedBandit
from .bitflipping import BitFlipping
from .cartpole import CartPole
from .energy import EnergyMicrogrid
from .grid_world import GridWorld
from .gym import GymAdapter, convert_space, make_atari, make_gym, make_gym_vec, make_minigrid
from .inventory import InventoryManagement
from .lunar_lander import LunarLander
from .mountain_car import MountainCar, MountainCarContinuous
from .navigation import Navigation2D
from .pendulum import Pendulum
from .point_mass import PointMass
from .portfolio import PortfolioAllocation
from .pricing import DynamicPricing
from .queueing import QueueAdmissionControl
from .reacher import ReacherArm
from .supply_chain import SupplyChain
from .thermostat import Thermostat

__all__ = [
    # classic / toy
    "GridWorld",
    "BitFlipping",
    "MultiArmedBandit",
    "BernoulliBandit",
    "CartPole",
    "Pendulum",
    "PointMass",
    "MountainCar",
    "MountainCarContinuous",
    "Acrobot",
    # complex / varied scenarios
    "ReacherArm",
    "Navigation2D",
    "LunarLander",
    "PortfolioAllocation",
    # applied (operational / business decisions)
    "InventoryManagement",
    "Thermostat",
    "DynamicPricing",
    "QueueAdmissionControl",
    "EnergyMicrogrid",
    "SupplyChain",
    # gymnasium interop
    "GymAdapter",
    "make_gym",
    "make_gym_vec",
    "make_atari",
    "make_minigrid",
    "convert_space",
]
