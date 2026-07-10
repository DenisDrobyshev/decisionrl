"""Swarm-intelligence / nature-inspired optimizers.

Continuous minimizers (ask/tell like the rest):

* :class:`PSO` - Particle Swarm Optimization.
* :class:`FireflyAlgorithm` - fireflies attracted to brighter neighbours.
* :class:`ArtificialBeeColony` - employed + scout bees.
* :class:`GreyWolfOptimizer` - alpha/beta/delta leader hierarchy.
* :class:`BatAlgorithm` - echolocation with loudness / pulse-rate adaptation.

Combinatorial:

* :class:`AntColonyTSP` - Ant Colony Optimization for the Travelling Salesperson
  problem (with :func:`random_cities` / :func:`distance_matrix` helpers).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .base import BlackBoxOptimizer

__all__ = [
    "PSO",
    "FireflyAlgorithm",
    "ArtificialBeeColony",
    "GreyWolfOptimizer",
    "BatAlgorithm",
    "AntColonyTSP",
    "random_cities",
    "distance_matrix",
]


class PSO(BlackBoxOptimizer):
    def __init__(self, dim, popsize=40, w=0.7, c1=1.5, c2=1.5, init_std=1.0, bounds=None, seed=None):
        super().__init__(dim, popsize, bounds, seed)
        self.w, self.c1, self.c2 = float(w), float(c1), float(c2)
        self.pos = self._init_pop(popsize, init_std)
        self.vel = np.zeros((popsize, dim))
        self.pbest = self.pos.copy()
        self.pbest_f = np.full(popsize, np.inf)
        self.gbest = self.pos[0].copy()
        self.gbest_f = np.inf
        self._init = False

    def ask(self):
        if not self._init:
            return self.pos.copy()
        r1 = self.rng.random((self.popsize, self.dim))
        r2 = self.rng.random((self.popsize, self.dim))
        self.vel = self.w * self.vel + self.c1 * r1 * (self.pbest - self.pos) + self.c2 * r2 * (self.gbest - self.pos)
        self.pos = self._clip(self.pos + self.vel)
        return self.pos.copy()

    def tell(self, population, fitnesses):
        self._track_best(population, fitnesses)
        self.pos = population
        improved = fitnesses < self.pbest_f
        self.pbest[improved] = population[improved]
        self.pbest_f[improved] = fitnesses[improved]
        i = int(np.argmin(fitnesses))
        if fitnesses[i] < self.gbest_f:
            self.gbest_f = float(fitnesses[i])
            self.gbest = population[i].copy()
        self._init = True


class FireflyAlgorithm(BlackBoxOptimizer):
    def __init__(self, dim, popsize=30, alpha=0.2, beta0=1.0, gamma=1.0, alpha_decay=0.97,
                 init_std=1.0, bounds=None, seed=None):
        super().__init__(dim, popsize, bounds, seed)
        self.beta0, self.alpha_decay = float(beta0), float(alpha_decay)
        # Scale the randomness and light absorption to the search domain so the
        # algorithm behaves well regardless of the variable ranges.
        self._scale = float(np.mean(self.high - self.low)) if self.low is not None else 1.0
        self.alpha = float(alpha) * self._scale
        self.gamma = float(gamma) / (self._scale**2)
        self.pop = self._init_pop(popsize, init_std)
        self.fit = None
        self._init = False

    def ask(self):
        if not self._init:
            return self.pop.copy()
        new = self.pop.copy()
        for i in range(self.popsize):
            for j in range(self.popsize):
                if self.fit[j] < self.fit[i]:
                    r2 = np.sum((self.pop[i] - self.pop[j]) ** 2)
                    beta = self.beta0 * np.exp(-self.gamma * r2)
                    new[i] += beta * (self.pop[j] - new[i]) + self.alpha * (self.rng.random(self.dim) - 0.5)
        return self._clip(new)

    def tell(self, population, fitnesses):
        self._track_best(population, fitnesses)
        self.pop, self.fit = population.copy(), fitnesses.copy()
        self.alpha *= self.alpha_decay
        self._init = True


class ArtificialBeeColony(BlackBoxOptimizer):
    def __init__(self, dim, popsize=40, limit=20, init_std=1.0, bounds=None, seed=None):
        super().__init__(dim, popsize, bounds, seed)
        self.limit = int(limit)
        self.pop = self._init_pop(popsize, init_std)
        self.fit = None
        self.trials = np.zeros(popsize, dtype=int)
        self._init = False

    def ask(self):
        if not self._init:
            return self.pop.copy()
        cand = self.pop.copy()
        for i in range(self.popsize):
            k = self.rng.integers(self.popsize - 1)
            k = k + 1 if k >= i else k  # k != i
            phi = self.rng.uniform(-1, 1, self.dim)
            cand[i] = self.pop[i] + phi * (self.pop[i] - self.pop[k])
        return self._clip(cand)

    def tell(self, population, fitnesses):
        self._track_best(population, fitnesses)
        if not self._init:
            self.fit = fitnesses.copy()
            self._init = True
            return
        improved = fitnesses < self.fit
        self.pop[improved] = population[improved]
        self.fit[improved] = fitnesses[improved]
        self.trials[improved] = 0
        self.trials[~improved] += 1
        # scouts: abandon exhausted sources
        scouts = self.trials > self.limit
        if np.any(scouts):
            self.pop[scouts] = self._init_pop(int(scouts.sum()))
            self.fit[scouts] = np.inf
            self.trials[scouts] = 0


class GreyWolfOptimizer(BlackBoxOptimizer):
    def __init__(self, dim, popsize=30, iters=200, init_std=1.0, bounds=None, seed=None):
        super().__init__(dim, popsize, bounds, seed)
        self.max_iter = int(iters)
        self.pop = self._init_pop(popsize, init_std)
        self.alpha = self.beta = self.delta = self.pop[0].copy()
        self.t = 0
        self._init = False

    def ask(self):
        if not self._init:
            return self.pop.copy()
        a = max(0.0, 2 - 2 * self.t / self.max_iter)
        new = np.empty_like(self.pop)
        for i in range(self.popsize):
            xs = []
            for leader in (self.alpha, self.beta, self.delta):
                r1, r2 = self.rng.random(self.dim), self.rng.random(self.dim)
                A, C = 2 * a * r1 - a, 2 * r2
                D = np.abs(C * leader - self.pop[i])
                xs.append(leader - A * D)
            new[i] = np.mean(xs, axis=0)
        return self._clip(new)

    def tell(self, population, fitnesses):
        self._track_best(population, fitnesses)
        order = np.argsort(fitnesses)
        self.alpha = population[order[0]].copy()
        self.beta = population[order[1 % self.popsize]].copy()
        self.delta = population[order[2 % self.popsize]].copy()
        self.pop = population
        self.t += 1
        self._init = True


class BatAlgorithm(BlackBoxOptimizer):
    def __init__(self, dim, popsize=30, fmin=0.0, fmax=1.0, loudness=1.0, pulse=0.5,
                 alpha=0.9, gamma=0.9, inertia=0.9, init_std=1.0, bounds=None, seed=None):
        super().__init__(dim, popsize, bounds, seed)
        self.fmin, self.fmax = float(fmin), float(fmax)
        self.alpha, self.gamma, self.inertia = float(alpha), float(gamma), float(inertia)
        self._scale = float(np.mean(self.high - self.low)) if self.low is not None else 1.0
        self.A = np.full(popsize, float(loudness))
        self.r0 = float(pulse)
        self.r = np.full(popsize, float(pulse))
        self.pop = self._init_pop(popsize, init_std)
        self.vel = np.zeros((popsize, dim))
        self.fit = None
        self.best = self.pop[0].copy()
        self.t = 0
        self._init = False

    def ask(self):
        if not self._init:
            return self.pop.copy()
        freq = self.fmin + (self.fmax - self.fmin) * self.rng.random(self.popsize)
        # Damped velocity toward the current best (inertia keeps it from diverging).
        self.vel = self.inertia * self.vel + (self.pop - self.best) * freq[:, None]
        cand = self.pop + self.vel
        local = self.rng.random(self.popsize) > self.r
        n = int(local.sum())
        if n:
            step = 0.05 * self._scale * (self.A.mean() + 1e-8)
            cand[local] = self.best + step * self.rng.standard_normal((n, self.dim))
        self._cand = self._clip(cand)
        return self._cand

    def tell(self, population, fitnesses):
        self._track_best(population, fitnesses)
        if not self._init:
            self.fit = fitnesses.copy()
            self.best = population[int(np.argmin(fitnesses))].copy()
            self._init = True
            return
        accept = (fitnesses <= self.fit) & (self.rng.random(self.popsize) < self.A)
        self.pop[accept] = population[accept]
        self.fit[accept] = fitnesses[accept]
        self.A[accept] *= self.alpha
        self.t += 1
        self.r[accept] = self.r0 * (1 - np.exp(-self.gamma * self.t))
        i = int(np.argmin(self.fit))
        self.best = self.pop[i].copy()


# --------------------------------------------------------------------------- #
# Ant Colony Optimization for the Travelling Salesperson Problem.
# --------------------------------------------------------------------------- #

def random_cities(n: int, seed: Optional[int] = None) -> np.ndarray:
    return np.random.default_rng(seed).random((n, 2))


def distance_matrix(cities: np.ndarray) -> np.ndarray:
    diff = cities[:, None, :] - cities[None, :, :]
    return np.sqrt((diff**2).sum(-1))


class AntColonyTSP:
    """Ant System for the TSP: pheromone-guided tour construction."""

    def __init__(self, distances, n_ants=None, alpha=1.0, beta=5.0, evaporation=0.5,
                 q=1.0, seed=None):
        self.d = np.asarray(distances, dtype=np.float64)
        self.n = self.d.shape[0]
        self.n_ants = int(n_ants) if n_ants else self.n
        self.alpha, self.beta, self.rho, self.q = alpha, beta, evaporation, q
        self.rng = np.random.default_rng(seed)
        with np.errstate(divide="ignore"):
            self.eta = np.where(self.d > 0, 1.0 / self.d, 0.0)  # visibility
        self.tau = np.ones((self.n, self.n))
        self.best_tour: Optional[np.ndarray] = None
        self.best_len = np.inf

    def _tour_length(self, tour):
        return float(self.d[tour, np.roll(tour, -1)].sum())

    def _build_tour(self):
        start = int(self.rng.integers(self.n))
        tour = [start]
        unvisited = set(range(self.n)) - {start}
        current = start
        while unvisited:
            cities = np.array(sorted(unvisited))
            w = (self.tau[current, cities] ** self.alpha) * (self.eta[current, cities] ** self.beta)
            total = w.sum()
            probs = w / total if total > 0 else np.full(len(cities), 1.0 / len(cities))
            nxt = int(self.rng.choice(cities, p=probs))
            tour.append(nxt)
            unvisited.discard(nxt)
            current = nxt
        return np.array(tour)

    def solve(self, iters: int = 100) -> Tuple[np.ndarray, float, np.ndarray]:
        history = np.empty(iters)
        for it in range(iters):
            tours = [self._build_tour() for _ in range(self.n_ants)]
            lengths = np.array([self._tour_length(t) for t in tours])
            best = int(np.argmin(lengths))
            if lengths[best] < self.best_len:
                self.best_len = float(lengths[best])
                self.best_tour = tours[best].copy()
            self.tau *= 1 - self.rho  # evaporation
            for tour, length in zip(tours, lengths):
                deposit = self.q / length
                a, b = tour, np.roll(tour, -1)
                self.tau[a, b] += deposit
                self.tau[b, a] += deposit
            history[it] = self.best_len
        return self.best_tour, self.best_len, history
