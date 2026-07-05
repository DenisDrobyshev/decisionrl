"""Scalar schedules for exploration rates, learning rates, etc."""

from __future__ import annotations

__all__ = ["Schedule", "ConstantSchedule", "LinearSchedule", "ExponentialSchedule"]


class Schedule:
    """A callable mapping a training progress step to a scalar value."""

    def value(self, step: int) -> float:  # pragma: no cover - overridden
        raise NotImplementedError

    def __call__(self, step: int) -> float:
        return self.value(step)


class ConstantSchedule(Schedule):
    def __init__(self, value: float) -> None:
        self._value = float(value)

    def value(self, step: int) -> float:
        return self._value


class LinearSchedule(Schedule):
    """Linearly anneal from ``start`` to ``end`` over ``duration`` steps.

    After ``duration`` steps the value is clamped to ``end``. This is the classic
    epsilon schedule used by DQN.
    """

    def __init__(self, start: float, end: float, duration: int) -> None:
        assert duration > 0, "duration must be positive"
        self.start = float(start)
        self.end = float(end)
        self.duration = int(duration)

    def value(self, step: int) -> float:
        frac = min(max(step / self.duration, 0.0), 1.0)
        return self.start + frac * (self.end - self.start)


class ExponentialSchedule(Schedule):
    """Multiplicative decay: ``value = max(end, start * decay ** step)``."""

    def __init__(self, start: float, end: float, decay: float) -> None:
        assert 0.0 < decay <= 1.0, "decay must be in (0, 1]"
        self.start = float(start)
        self.end = float(end)
        self.decay = float(decay)

    def value(self, step: int) -> float:
        return max(self.end, self.start * (self.decay ** step))
