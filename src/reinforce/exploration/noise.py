"""Action-space noise processes for exploration in continuous control."""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["ActionNoise", "GaussianNoise", "OrnsteinUhlenbeckNoise"]


class ActionNoise:
    """Base class for action noise processes."""

    def reset(self) -> None:
        """Reset any internal state (called at episode boundaries)."""

    def __call__(self) -> np.ndarray:  # pragma: no cover - overridden
        raise NotImplementedError


class GaussianNoise(ActionNoise):
    """Uncorrelated Gaussian noise, the default choice for TD3/DDPG/SAC."""

    def __init__(self, mean: np.ndarray, sigma: np.ndarray, seed: Optional[int] = None) -> None:
        self.mean = np.asarray(mean, dtype=np.float32)
        self.sigma = np.asarray(sigma, dtype=np.float32)
        self._rng = np.random.default_rng(seed)

    def __call__(self) -> np.ndarray:
        return self._rng.normal(self.mean, self.sigma).astype(np.float32)


class OrnsteinUhlenbeckNoise(ActionNoise):
    """Temporally correlated OU noise (used by the original DDPG paper).

    Useful for environments with momentum/inertia where correlated exploration
    is more effective than white noise.
    """

    def __init__(
        self,
        mean: np.ndarray,
        sigma: np.ndarray,
        theta: float = 0.15,
        dt: float = 1e-2,
        initial: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.mean = np.asarray(mean, dtype=np.float32)
        self.sigma = np.asarray(sigma, dtype=np.float32)
        self.theta = float(theta)
        self.dt = float(dt)
        self._initial = initial
        self._rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        self._prev = (
            np.array(self._initial, dtype=np.float32)
            if self._initial is not None
            else np.zeros_like(self.mean)
        )

    def __call__(self) -> np.ndarray:
        noise = self._rng.normal(size=self.mean.shape)
        x = (
            self._prev
            + self.theta * (self.mean - self._prev) * self.dt
            + self.sigma * np.sqrt(self.dt) * noise
        )
        self._prev = x.astype(np.float32)
        return self._prev
