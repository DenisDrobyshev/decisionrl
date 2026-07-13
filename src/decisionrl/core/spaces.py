"""Minimal, dependency-free observation/action spaces.

These mirror the attribute names of :mod:`gymnasium.spaces` (``n``, ``shape``,
``low``, ``high``, ``dtype``) so that agents written against this library work
transparently with Gymnasium environments *and* with the built-in ones, without
importing Gymnasium.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

__all__ = ["Space", "Discrete", "Box", "Dict", "is_discrete", "flatdim", "flatten"]


class Space:
    """Base class for spaces."""

    def __init__(self, shape: Optional[Sequence[int]], dtype: np.dtype) -> None:
        self.shape: Optional[Tuple[int, ...]] = None if shape is None else tuple(int(x) for x in shape)
        self.dtype = np.dtype(dtype)
        self._rng = np.random.default_rng()

    def seed(self, seed: Optional[int] = None) -> None:
        """Seed the space's internal RNG used by :meth:`sample`."""
        self._rng = np.random.default_rng(seed)

    def sample(self) -> np.ndarray:  # pragma: no cover - overridden
        raise NotImplementedError

    def contains(self, x) -> bool:  # pragma: no cover - overridden
        raise NotImplementedError

    def __contains__(self, x) -> bool:
        return self.contains(x)


class Discrete(Space):
    """A finite set of integers ``{start, ..., start + n - 1}``."""

    def __init__(self, n: int, start: int = 0) -> None:
        assert n > 0, "n (number of elements) must be positive"
        super().__init__((), np.int64)
        self.n = int(n)
        self.start = int(start)

    def sample(self) -> int:
        return int(self.start + self._rng.integers(self.n))

    def contains(self, x) -> bool:
        if isinstance(x, (np.generic, np.ndarray)):
            if x.shape != ():
                return False
            x = int(x)
        if not isinstance(x, (int, np.integer)):
            return False
        return self.start <= int(x) < self.start + self.n

    def __repr__(self) -> str:
        if self.start == 0:
            return f"Discrete({self.n})"
        return f"Discrete({self.n}, start={self.start})"

    def __eq__(self, other) -> bool:
        return isinstance(other, Discrete) and other.n == self.n and other.start == self.start


class Box(Space):
    """A (possibly bounded) box in R^n."""

    def __init__(
        self,
        low: Union[float, Sequence[float], np.ndarray],
        high: Union[float, Sequence[float], np.ndarray],
        shape: Optional[Sequence[int]] = None,
        dtype: np.dtype = np.float32,
    ) -> None:
        dtype = np.dtype(dtype)
        if shape is None:
            if np.isscalar(low) and np.isscalar(high):
                raise ValueError("shape must be provided when low and high are scalars")
            shape = np.broadcast(np.asarray(low), np.asarray(high)).shape
        shape = tuple(int(x) for x in shape)
        self.low = np.broadcast_to(np.asarray(low, dtype=dtype), shape).astype(dtype, copy=True)
        self.high = np.broadcast_to(np.asarray(high, dtype=dtype), shape).astype(dtype, copy=True)
        super().__init__(shape, dtype)

    @property
    def bounded_below(self) -> np.ndarray:
        return np.isfinite(self.low)

    @property
    def bounded_above(self) -> np.ndarray:
        return np.isfinite(self.high)

    def sample(self) -> np.ndarray:
        # Sample uniformly within finite bounds; fall back to a standard normal
        # for unbounded dimensions (matching Gymnasium's behaviour closely).
        sample = np.empty(self.shape, dtype=np.float64)  # type: ignore[type-var]
        both = self.bounded_below & self.bounded_above
        low_only = self.bounded_below & ~self.bounded_above
        high_only = ~self.bounded_below & self.bounded_above
        free = ~self.bounded_below & ~self.bounded_above

        sample[free] = self._rng.normal(size=free.sum())
        sample[both] = self._rng.uniform(low=self.low[both], high=self.high[both])
        sample[low_only] = self.low[low_only] + self._rng.exponential(size=low_only.sum())
        sample[high_only] = self.high[high_only] - self._rng.exponential(size=high_only.sum())
        return sample.astype(self.dtype)  # type: ignore[return-value]

    def contains(self, x) -> bool:
        x = np.asarray(x)
        return bool(
            x.shape == self.shape
            and np.all(x >= self.low)
            and np.all(x <= self.high)
        )

    def __repr__(self) -> str:
        return f"Box({self.low.min()}, {self.high.max()}, {self.shape}, {self.dtype.name})"

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Box)
            and other.shape == self.shape
            and np.allclose(other.low, self.low)
            and np.allclose(other.high, self.high)
        )


class Dict(Space):
    """A dictionary of named subspaces (for multi-modal observations).

    Not consumed directly by the networks; wrap the environment in
    :class:`~decisionrl.wrappers.FlattenDictObservation` to feed agents a single
    flattened Box.
    """

    def __init__(self, spaces: dict) -> None:
        super().__init__(None, np.float32)
        self.spaces = dict(spaces)

    def seed(self, seed: Optional[int] = None) -> None:
        for i, s in enumerate(self.spaces.values()):
            s.seed(None if seed is None else seed + i)

    def sample(self) -> dict:
        return {k: s.sample() for k, s in self.spaces.items()}

    def contains(self, x) -> bool:
        return isinstance(x, dict) and set(x) == set(self.spaces) and all(
            self.spaces[k].contains(v) for k, v in x.items()
        )

    def __repr__(self) -> str:
        return "Dict(" + ", ".join(f"{k}: {v}" for k, v in self.spaces.items()) + ")"


def is_discrete(space: Space) -> bool:
    """Return ``True`` for discrete action/observation spaces.

    Works with both this library's spaces and ``gymnasium.spaces`` by
    duck-typing on the presence of an integer ``n`` attribute without a
    continuous ``low``/``high`` pair.
    """
    return hasattr(space, "n") and not hasattr(space, "low")


def flatdim(space: Space) -> int:
    """Number of scalar components needed to represent one element of ``space``."""
    if isinstance(space, Dict):
        return sum(flatdim(s) for s in space.spaces.values())
    if is_discrete(space):
        return int(space.n)  # one-hot dimensionality
    return int(np.prod(space.shape))


def flatten(space: Space, x) -> np.ndarray:
    """Flatten a sample ``x`` of ``space`` into a 1-D float32 vector.

    Discrete values become one-hot; Box values are raveled; Dict values are the
    concatenation of their flattened subspaces (in insertion order).
    """
    if isinstance(space, Dict):
        return np.concatenate([flatten(s, x[k]) for k, s in space.spaces.items()])
    if is_discrete(space):
        vec = np.zeros(int(space.n), dtype=np.float32)
        vec[int(x)] = 1.0
        return vec
    return np.asarray(x, dtype=np.float32).ravel()
