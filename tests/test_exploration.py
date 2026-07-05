import numpy as np
import pytest

from reinforce.exploration import (
    ConstantSchedule,
    ExponentialSchedule,
    GaussianNoise,
    LinearSchedule,
    OrnsteinUhlenbeckNoise,
)


def test_constant_schedule():
    s = ConstantSchedule(0.3)
    assert s(0) == 0.3 and s(1000) == 0.3


def test_linear_schedule_endpoints_and_clamp():
    s = LinearSchedule(1.0, 0.1, duration=100)
    assert s(0) == pytest.approx(1.0)
    assert s(50) == pytest.approx(0.55)
    assert s(100) == pytest.approx(0.1)
    assert s(500) == pytest.approx(0.1)  # clamped past duration


def test_exponential_schedule_floor():
    s = ExponentialSchedule(1.0, 0.1, decay=0.9)
    assert s(0) == pytest.approx(1.0)
    assert s(1) == pytest.approx(0.9)
    assert s(1000) == pytest.approx(0.1)  # never drops below end


def test_gaussian_noise_shape_and_seed():
    a = GaussianNoise(np.zeros(3), np.ones(3) * 0.2, seed=0)
    b = GaussianNoise(np.zeros(3), np.ones(3) * 0.2, seed=0)
    na, nb = a(), b()
    assert na.shape == (3,)
    np.testing.assert_allclose(na, nb)


def test_ou_noise_reset_and_shape():
    ou = OrnsteinUhlenbeckNoise(np.zeros(2), np.ones(2) * 0.2, seed=0)
    first = ou()
    ou()
    ou.reset()
    assert np.allclose(ou._prev, 0.0)
    assert first.shape == (2,)
