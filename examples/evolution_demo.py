"""Evolutionary & swarm optimization demos + README figures.

Generates three figures in docs/assets/:

    evolution_benchmark.png       convergence of 8 optimizers on Rastrigin
    evolution_neuroevolution.png  neuroevolution solving CartPole (gradient-free)
    evolution_aco_tsp.png         Ant Colony Optimization tour for a TSP

Run: python examples/evolution_demo.py
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from reinforce.envs import CartPole
from reinforce.evolution import (
    CEM,
    CMAES,
    PSO,
    AntColonyTSP,
    ArtificialBeeColony,
    DifferentialEvolution,
    FireflyAlgorithm,
    GeneticAlgorithm,
    GreyWolfOptimizer,
    NeuroevolutionAgent,
    distance_matrix,
    minimize,
    random_cities,
)
from reinforce.evolution.functions import rastrigin
from reinforce.utils import Logger, set_seed

ASSETS = os.path.join(os.path.dirname(__file__), "..", "docs", "assets")
os.makedirs(ASSETS, exist_ok=True)
PALETTE = ["#2563eb", "#db2777", "#16a34a", "#f59e0b", "#7c3aed", "#0891b2", "#dc2626", "#65a30d"]


def benchmark_convergence() -> None:
    dim, bounds, iters = 10, (-5.12, 5.12), 200
    optimizers = {
        "CEM": CEM(dim, bounds=bounds, seed=0),
        "CMA-ES": CMAES(dim, bounds=bounds, seed=0),
        "DE": DifferentialEvolution(dim, bounds=bounds, seed=0),
        "GA": GeneticAlgorithm(dim, bounds=bounds, seed=0),
        "PSO": PSO(dim, bounds=bounds, seed=0),
        "Firefly": FireflyAlgorithm(dim, bounds=bounds, seed=0),
        "ABC": ArtificialBeeColony(dim, bounds=bounds, seed=0),
        "GWO": GreyWolfOptimizer(dim, bounds=bounds, iters=iters, seed=0),
    }
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=110)
    for (name, opt), color in zip(optimizers.items(), PALETTE):
        _, _, history = minimize(rastrigin, opt, iters)
        ax.plot(history + 1e-9, label=name, color=color, lw=2)
    ax.set_yscale("log")
    ax.set_xlabel("generation")
    ax.set_ylabel("best objective (Rastrigin, log scale)")
    ax.set_title(f"Gradient-free optimizers on Rastrigin ({dim}-D)")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "evolution_benchmark.png"))
    plt.close(fig)
    print("wrote evolution_benchmark.png")


def neuroevolution_cartpole() -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=110)
    for opt, color in zip(["cem", "cmaes", "ga", "pso"], PALETTE):
        set_seed(0)
        agent = NeuroevolutionAgent(CartPole(), optimizer=opt, hidden_sizes=(16,), popsize=24,
                                    seed=0, logger=Logger(verbose=0))
        agent.learn(40_000)
        steps, returns = zip(*agent.history_)
        ax.plot(steps, returns, label=opt.upper(), color=color, lw=2)
    ax.axhline(500, ls="--", color="#94a3b8", label="solved (500)")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("best episode return")
    ax.set_title("Neuroevolution on CartPole (no gradients)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "evolution_neuroevolution.png"))
    plt.close(fig)
    print("wrote evolution_neuroevolution.png")


def aco_tsp() -> None:
    cities = random_cities(20, seed=3)
    d = distance_matrix(cities)
    aco = AntColonyTSP(d, seed=0)
    tour, length, history = aco.solve(120)
    loop = np.append(tour, tour[0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), dpi=110)
    ax1.plot(cities[loop, 0], cities[loop, 1], "-o", color="#2563eb", ms=5)
    ax1.scatter(cities[:, 0], cities[:, 1], color="#db2777", zorder=3, s=25)
    ax1.set_title(f"ACO best tour (length {length:.2f})")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.plot(history, color="#16a34a", lw=2)
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("best tour length")
    ax2.set_title("ACO convergence")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "evolution_aco_tsp.png"))
    plt.close(fig)
    print("wrote evolution_aco_tsp.png")


if __name__ == "__main__":
    benchmark_convergence()
    neuroevolution_cartpole()
    aco_tsp()
