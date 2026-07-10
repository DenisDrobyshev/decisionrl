"""Gradient-free optimization: evolutionary strategies and swarm intelligence.

A unified **ask/tell** family of black-box optimizers plus a neuroevolution agent
that trains policies without gradients.

Evolutionary
    :class:`CEM`, :class:`CMAES`, :class:`DifferentialEvolution`,
    :class:`GeneticAlgorithm`, :class:`OpenAIES`, :class:`ARS`,
    :class:`SimulatedAnnealing`.
Swarm
    :class:`PSO`, :class:`FireflyAlgorithm`, :class:`ArtificialBeeColony`,
    :class:`GreyWolfOptimizer`, :class:`BatAlgorithm`, :class:`AntColonyTSP`.
Neuroevolution
    :class:`NeuroevolutionAgent` - train an RL policy with any optimizer above.

    >>> from reinforce.evolution import CEM, minimize
    >>> from reinforce.evolution.functions import rastrigin
    >>> x, f, hist = minimize(rastrigin, CEM(dim=10, seed=0), iters=200)
"""

from .base import BlackBoxOptimizer, minimize
from .evolutionary import (
    ARS,
    CEM,
    CMAES,
    DifferentialEvolution,
    GeneticAlgorithm,
    OpenAIES,
    SimulatedAnnealing,
)
from .functions import BENCHMARKS, ackley, griewank, rastrigin, rosenbrock, sphere
from .neuroevolution import OPTIMIZERS, NeuroevolutionAgent
from .swarm import (
    PSO,
    AntColonyTSP,
    ArtificialBeeColony,
    BatAlgorithm,
    FireflyAlgorithm,
    GreyWolfOptimizer,
    distance_matrix,
    random_cities,
)

__all__ = [
    "BlackBoxOptimizer",
    "minimize",
    # evolutionary
    "CEM",
    "CMAES",
    "DifferentialEvolution",
    "GeneticAlgorithm",
    "OpenAIES",
    "ARS",
    "SimulatedAnnealing",
    # swarm
    "PSO",
    "FireflyAlgorithm",
    "ArtificialBeeColony",
    "GreyWolfOptimizer",
    "BatAlgorithm",
    "AntColonyTSP",
    "random_cities",
    "distance_matrix",
    # neuroevolution
    "NeuroevolutionAgent",
    "OPTIMIZERS",
    # benchmark functions
    "BENCHMARKS",
    "sphere",
    "rastrigin",
    "ackley",
    "rosenbrock",
    "griewank",
]
