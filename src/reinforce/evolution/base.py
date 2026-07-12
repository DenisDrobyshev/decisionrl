"""Common interface for gradient-free (black-box) optimizers.

Every optimizer follows the classic **ask / tell** protocol so they are
interchangeable and easy to benchmark:

    opt = CEM(dim=10, popsize=64, seed=0)
    for _ in range(200):
        pop = opt.ask()                      # (popsize, dim) candidate solutions
        fit = np.array([f(x) for x in pop])  # evaluate (lower = better)
        opt.tell(pop, fit)
    x_best, f_best = opt.best_x, opt.best_f

All optimizers **minimize** the objective. Use :func:`minimize` for the loop.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

__all__ = ["BlackBoxOptimizer", "minimize"]


class BlackBoxOptimizer:
    """Base class for population-based black-box minimizers."""

    def __init__(
        self,
        dim: int,
        popsize: int,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.dim = int(dim)
        self.popsize = int(popsize)
        self.rng = np.random.default_rng(seed)
        if bounds is not None:
            low, high = bounds
            self.low = np.broadcast_to(np.asarray(low, dtype=np.float64), (self.dim,)).copy()
            self.high = np.broadcast_to(np.asarray(high, dtype=np.float64), (self.dim,)).copy()
        else:
            self.low = self.high = None
        self.best_x: Optional[np.ndarray] = None
        self.best_f: float = np.inf

    def _clip(self, x: np.ndarray) -> np.ndarray:
        if self.low is None:
            return x
        return np.clip(x, self.low, self.high)

    def _init_pop(self, n: int, init_std: float = 1.0) -> np.ndarray:
        """Initialize ``n`` solutions: uniform in bounds, or N(0, init_std)."""
        if self.low is not None:
            return self.rng.uniform(self.low, self.high, size=(n, self.dim))
        return self.rng.standard_normal((n, self.dim)) * init_std

    def _track_best(self, population: np.ndarray, fitnesses: np.ndarray) -> None:
        i = int(np.argmin(fitnesses))
        if fitnesses[i] < self.best_f:
            self.best_f = float(fitnesses[i])
            self.best_x = population[i].copy()

    def ask(self) -> np.ndarray:  # pragma: no cover - overridden
        raise NotImplementedError

    def tell(self, population: np.ndarray, fitnesses: np.ndarray) -> None:  # pragma: no cover
        raise NotImplementedError


def minimize(
    fn: Callable[[np.ndarray], float],
    optimizer: BlackBoxOptimizer,
    iters: int,
    callback: Optional[Callable[[int, float], None]] = None,
    batched: bool = False,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Run ``optimizer`` on ``fn`` for ``iters`` generations.

    With ``batched=True`` the whole population ``(popsize, dim)`` is passed to
    ``fn`` in one call (expecting a ``(popsize,)`` result) — much faster for
    vectorized objectives. Returns ``(best_x, best_f, history)`` where
    ``history[t]`` is the best objective value through generation ``t``.
    """
    history = np.empty(iters, dtype=np.float64)
    for t in range(iters):
        population = optimizer.ask()
        if batched:
            fitnesses = np.asarray(fn(population), dtype=np.float64).reshape(-1)
        else:
            fitnesses = np.array([float(fn(x)) for x in population], dtype=np.float64)
        optimizer.tell(population, fitnesses)
        history[t] = optimizer.best_f
        if callback is not None:
            callback(t, optimizer.best_f)
    return optimizer.best_x, optimizer.best_f, history
