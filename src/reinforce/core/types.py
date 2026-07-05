"""Core data types shared across the library."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

__all__ = ["Transition"]


@dataclass
class Transition:
    """A single environment transition.

    ``done`` follows the *bootstrapping* convention: it is ``True`` only when the
    episode ended because the MDP reached a terminal state (Gymnasium's
    ``terminated``), **not** when it was cut off by a time limit
    (``truncated``). This distinction is what makes value bootstrapping correct.
    """

    obs: np.ndarray
    action: Any
    reward: float
    next_obs: np.ndarray
    done: bool
    info: Dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.info is None:
            self.info = {}
