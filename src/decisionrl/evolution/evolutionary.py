"""Evolutionary / evolution-strategy black-box optimizers (all minimize).

* :class:`CEM` - Cross-Entropy Method.
* :class:`CMAES` - Covariance Matrix Adaptation Evolution Strategy.
* :class:`DifferentialEvolution` - DE/rand/1/bin.
* :class:`GeneticAlgorithm` - real-coded GA (tournament + blend + Gaussian mutation).
* :class:`OpenAIES` - Natural Evolution Strategy with mirrored sampling.
* :class:`ARS` - Augmented Random Search (V1-t).
* :class:`SimulatedAnnealing` - parallel-chain simulated annealing.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import BlackBoxOptimizer

__all__ = [
    "CEM",
    "CMAES",
    "DifferentialEvolution",
    "GeneticAlgorithm",
    "OpenAIES",
    "ARS",
    "SimulatedAnnealing",
]


class CEM(BlackBoxOptimizer):
    def __init__(self, dim, popsize=64, elite_frac=0.2, init_std=1.0, mean=None,
                 extra_std=0.0, bounds=None, seed=None):
        super().__init__(dim, popsize, bounds, seed)
        self.mean = np.zeros(dim) if mean is None else np.asarray(mean, dtype=np.float64).copy()
        self.std = np.full(dim, float(init_std))
        self.n_elite = max(1, int(elite_frac * popsize))
        self.extra_std = float(extra_std)

    def ask(self):
        pop = self.mean + self.std * self.rng.standard_normal((self.popsize, self.dim))
        return self._clip(pop)

    def tell(self, population, fitnesses):
        self._track_best(population, fitnesses)
        elite = population[np.argsort(fitnesses)[: self.n_elite]]
        self.mean = elite.mean(axis=0)
        self.std = elite.std(axis=0) + 1e-8 + self.extra_std


class CMAES(BlackBoxOptimizer):
    """(mu/mu_w, lambda)-CMA-ES (Hansen). Robust default black-box optimizer."""

    def __init__(self, dim, popsize=None, sigma=0.5, mean=None, bounds=None, seed=None):
        lam = int(popsize) if popsize else 4 + int(3 * np.log(dim))
        super().__init__(dim, lam, bounds, seed)
        n = dim
        self.sigma = float(sigma)
        self.xmean = np.zeros(n) if mean is None else np.asarray(mean, dtype=np.float64).copy()

        self.mu = lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = w / w.sum()
        self.mueff = 1.0 / np.sum(self.weights**2)

        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs

        self.pc = np.zeros(n)
        self.ps = np.zeros(n)
        self.C = np.eye(n)
        self.invsqrtC = np.eye(n)
        self.chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        self.counteval = 0
        self._z = None

    def ask(self):
        n = self.dim
        z = self.rng.standard_normal((self.popsize, n))
        eigvals, B = np.linalg.eigh(self.C)
        eigvals = np.clip(eigvals, 1e-20, None)
        D = np.sqrt(eigvals)
        y = (z * D) @ B.T
        self._B, self._D = B, D
        self._y = y
        return self._clip(self.xmean + self.sigma * y)

    def tell(self, population, fitnesses):
        self._track_best(population, fitnesses)
        n = self.dim
        self.counteval += self.popsize
        order = np.argsort(fitnesses)[: self.mu]
        y = self._y[order]
        xold = self.xmean.copy()
        yw = self.weights @ y
        self.xmean = xold + self.sigma * yw

        self.invsqrtC = self._B @ np.diag(1.0 / self._D) @ self._B.T
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * (self.invsqrtC @ yw)
        hsig = (np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * self.counteval / self.popsize))
                / self.chiN) < (1.4 + 2 / (n + 1))
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw

        artmp = y
        self.C = (
            (1 - self.c1 - self.cmu) * self.C
            + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
            + self.cmu * (artmp.T * self.weights) @ artmp
        )
        self.C = np.triu(self.C) + np.triu(self.C, 1).T
        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))


class DifferentialEvolution(BlackBoxOptimizer):
    def __init__(self, dim, popsize=40, F=0.8, CR=0.9, init_std=1.0, bounds=None, seed=None):
        super().__init__(dim, popsize, bounds, seed)
        self.F, self.CR = float(F), float(CR)
        self.pop = self._init_pop(popsize, init_std)
        self.fit = None
        self._init = False

    def ask(self):
        if not self._init:
            return self.pop.copy()
        trials = np.empty_like(self.pop)
        for i in range(self.popsize):
            idx = self.rng.choice(np.delete(np.arange(self.popsize), i), size=3, replace=False)
            a, b, c = self.pop[idx]
            mutant = a + self.F * (b - c)
            cross = self.rng.random(self.dim) < self.CR
            cross[self.rng.integers(self.dim)] = True
            trials[i] = self._clip(np.where(cross, mutant, self.pop[i]))
        return trials

    def tell(self, population, fitnesses):
        self._track_best(population, fitnesses)
        if not self._init:
            self.fit = fitnesses.copy()
            self._init = True
            return
        improved = fitnesses <= self.fit
        self.pop[improved] = population[improved]
        self.fit[improved] = fitnesses[improved]


class GeneticAlgorithm(BlackBoxOptimizer):
    def __init__(self, dim, popsize=50, elite_frac=0.1, tournament_k=3,
                 mutation_rate=0.1, mutation_std=0.3, init_std=1.0, bounds=None, seed=None):
        super().__init__(dim, popsize, bounds, seed)
        self.n_elite = max(1, int(elite_frac * popsize))
        self.k = int(tournament_k)
        self.mutation_rate = float(mutation_rate)
        self.mutation_std = float(mutation_std)
        self.pop = self._init_pop(popsize, init_std)
        self.fit = None
        self._init = False

    def _tournament(self):
        idx = self.rng.integers(0, self.popsize, size=self.k)
        return self.pop[idx[np.argmin(self.fit[idx])]]

    def ask(self):
        if not self._init:
            return self.pop.copy()
        order = np.argsort(self.fit)
        children = [self.pop[e].copy() for e in order[: self.n_elite]]
        while len(children) < self.popsize:
            p1, p2 = self._tournament(), self._tournament()
            alpha = self.rng.random(self.dim)
            child = alpha * p1 + (1 - alpha) * p2  # blend crossover
            mask = self.rng.random(self.dim) < self.mutation_rate
            child = child + mask * self.rng.normal(0, self.mutation_std, self.dim)
            children.append(self._clip(child))
        return np.asarray(children)

    def tell(self, population, fitnesses):
        self._track_best(population, fitnesses)
        self.pop, self.fit = population.copy(), fitnesses.copy()
        self._init = True


class OpenAIES(BlackBoxOptimizer):
    """Natural Evolution Strategy (Salimans et al., 2017) with mirrored sampling."""

    def __init__(self, dim, popsize=50, sigma=0.1, lr=0.05, mean=None, bounds=None, seed=None):
        popsize += popsize % 2  # even for mirrored pairs
        super().__init__(dim, popsize, bounds, seed)
        self.sigma, self.lr = float(sigma), float(lr)
        self.mean = np.zeros(dim) if mean is None else np.asarray(mean, dtype=np.float64).copy()

    def ask(self):
        half = self.popsize // 2
        eps = self.rng.standard_normal((half, self.dim))
        self._eps = np.concatenate([eps, -eps], axis=0)
        return self._clip(self.mean + self.sigma * self._eps)

    def tell(self, population, fitnesses):
        self._track_best(population, fitnesses)
        # minimize -> advantage is high when fitness is low
        adv = -(fitnesses - fitnesses.mean()) / (fitnesses.std() + 1e-8)
        grad = (self._eps.T @ adv) / (self.popsize * self.sigma)
        self.mean = self.mean + self.lr * grad


class ARS(BlackBoxOptimizer):
    """Augmented Random Search V1-t (Mania et al., 2018)."""

    def __init__(self, dim, popsize=32, step_size=0.05, noise_std=0.1, top_frac=0.5,
                 mean=None, bounds=None, seed=None):
        popsize += popsize % 2
        super().__init__(dim, popsize, bounds, seed)
        self.alpha, self.nu = float(step_size), float(noise_std)
        self.n_dir = self.popsize // 2
        self.n_top = max(1, int(top_frac * self.n_dir))
        self.mean = np.zeros(dim) if mean is None else np.asarray(mean, dtype=np.float64).copy()

    def ask(self):
        self._deltas = self.rng.standard_normal((self.n_dir, self.dim))
        plus = self.mean + self.nu * self._deltas
        minus = self.mean - self.nu * self._deltas
        return self._clip(np.concatenate([plus, minus], axis=0))

    def tell(self, population, fitnesses):
        self._track_best(population, fitnesses)
        r_plus = -fitnesses[: self.n_dir]  # rewards (maximize) = -loss
        r_minus = -fitnesses[self.n_dir:]
        best = np.argsort(np.maximum(r_plus, r_minus))[::-1][: self.n_top]
        sigma_r = np.std(np.concatenate([r_plus[best], r_minus[best]])) + 1e-8
        grad = ((r_plus[best] - r_minus[best])[:, None] * self._deltas[best]).sum(axis=0)
        self.mean = self.mean + (self.alpha / (self.n_top * sigma_r)) * grad


class SimulatedAnnealing(BlackBoxOptimizer):
    """Parallel-chain simulated annealing with geometric cooling."""

    def __init__(self, dim, popsize=20, temp=1.0, cooling=0.98, step=0.5,
                 init_std=1.0, bounds=None, seed=None):
        super().__init__(dim, popsize, bounds, seed)
        self.T = float(temp)
        self.cooling = float(cooling)
        self.step = float(step)
        self.current = self._init_pop(popsize, init_std)
        self.current_f: Optional[np.ndarray] = None
        self._init = False

    def ask(self):
        if not self._init:
            return self.current.copy()
        noise = self.rng.standard_normal((self.popsize, self.dim)) * self.step * max(self.T, 1e-3)
        return self._clip(self.current + noise)

    def tell(self, population, fitnesses):
        self._track_best(population, fitnesses)
        if not self._init:
            self.current, self.current_f = population.copy(), fitnesses.copy()
            self._init = True
            return
        delta = fitnesses - self.current_f
        accept = (delta < 0) | (self.rng.random(self.popsize) < np.exp(-delta / max(self.T, 1e-8)))
        self.current[accept] = population[accept]
        self.current_f[accept] = fitnesses[accept]
        self.T *= self.cooling
