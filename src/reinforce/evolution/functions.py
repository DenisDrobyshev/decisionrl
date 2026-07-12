"""Standard continuous optimization benchmark functions (global minimum 0).

Each is **batch-friendly**: it reduces over the last axis, so it accepts a single
vector ``(dim,)`` -> scalar *or* a whole population ``(pop, dim)`` -> ``(pop,)``.
The latter enables vectorized fitness evaluation (``minimize(..., batched=True)``).
``BENCHMARKS`` maps a name to ``(function, bounds, optimum_location)``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["sphere", "rastrigin", "ackley", "rosenbrock", "griewank", "BENCHMARKS"]


def sphere(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.sum(x**2, axis=-1)


def rastrigin(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[-1]
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=-1)


def ackley(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (
        -20 * np.exp(-0.2 * np.sqrt(np.mean(x**2, axis=-1)))
        - np.exp(np.mean(np.cos(2 * np.pi * x), axis=-1))
        + 20
        + np.e
    )


def rosenbrock(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.sum(100.0 * (x[..., 1:] - x[..., :-1] ** 2) ** 2 + (1 - x[..., :-1]) ** 2, axis=-1)


def griewank(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    i = np.arange(1, x.shape[-1] + 1)
    return np.sum(x**2, axis=-1) / 4000.0 - np.prod(np.cos(x / np.sqrt(i)), axis=-1) + 1


# name -> (function, (low, high), optimum location)
BENCHMARKS = {
    "sphere": (sphere, (-5.12, 5.12), 0.0),
    "rastrigin": (rastrigin, (-5.12, 5.12), 0.0),
    "ackley": (ackley, (-32.768, 32.768), 0.0),
    "rosenbrock": (rosenbrock, (-2.048, 2.048), 1.0),
    "griewank": (griewank, (-600.0, 600.0), 0.0),
}
