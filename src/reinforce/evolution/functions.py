"""Standard continuous optimization benchmark functions (global minimum 0).

Each returns a scalar for a 1-D input vector; ``BENCHMARKS`` maps a name to
``(function, bounds, optimum_location)`` for tests and demos.
"""

from __future__ import annotations

import numpy as np

__all__ = ["sphere", "rastrigin", "ackley", "rosenbrock", "griewank", "BENCHMARKS"]


def sphere(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sum(x**2))


def rastrigin(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(10 * x.size + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


def ackley(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    return float(
        -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / n))
        - np.exp(np.sum(np.cos(2 * np.pi * x)) / n)
        + 20
        + np.e
    )


def rosenbrock(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def griewank(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    i = np.arange(1, x.size + 1)
    return float(np.sum(x**2) / 4000.0 - np.prod(np.cos(x / np.sqrt(i))) + 1)


# name -> (function, (low, high), optimum location)
BENCHMARKS = {
    "sphere": (sphere, (-5.12, 5.12), 0.0),
    "rastrigin": (rastrigin, (-5.12, 5.12), 0.0),
    "ackley": (ackley, (-32.768, 32.768), 0.0),
    "rosenbrock": (rosenbrock, (-2.048, 2.048), 1.0),
    "griewank": (griewank, (-600.0, 600.0), 0.0),
}
