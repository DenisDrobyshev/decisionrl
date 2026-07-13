"""Online mean/variance tracking (Welford / Chan's parallel algorithm)."""

from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = ["RunningMeanStd"]


class RunningMeanStd:
    """Tracks a running mean and variance of a data stream.

    Used for observation and reward normalization. The update is numerically
    stable and works on batches, following the parallel variance algorithm.
    """

    def __init__(self, shape: Tuple[int, ...] = (), epsilon: float = 1e-4) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        self.var = m2 / tot_count
        self.count = tot_count

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var)

    def normalize(self, x: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
        return (np.asarray(x) - self.mean) / np.sqrt(self.var + epsilon)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: dict) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state["count"])
