import numpy as np
import pytest

from reinforce.core.spaces import Box, Discrete, flatdim, is_discrete


def test_discrete_basic():
    space = Discrete(5)
    assert space.n == 5
    assert space.shape == ()
    assert space.contains(0) and space.contains(4)
    assert not space.contains(5) and not space.contains(-1)
    assert 3 in space


def test_discrete_start():
    space = Discrete(3, start=2)
    assert space.contains(2) and space.contains(4)
    assert not space.contains(1) and not space.contains(5)


def test_discrete_sample_in_range_and_seeded():
    a = Discrete(10)
    a.seed(123)
    b = Discrete(10)
    b.seed(123)
    sa = [a.sample() for _ in range(50)]
    sb = [b.sample() for _ in range(50)]
    assert sa == sb
    assert all(0 <= x < 10 for x in sa)


def test_box_shapes_and_contains():
    space = Box(-1.0, 1.0, shape=(3,))
    assert space.shape == (3,)
    assert space.contains(np.zeros(3, dtype=np.float32))
    assert not space.contains(np.array([2.0, 0.0, 0.0], dtype=np.float32))
    assert not space.contains(np.zeros(2, dtype=np.float32))


def test_box_sample_within_bounds_and_seeded():
    space = Box(-2.0, 3.0, shape=(4,))
    space.seed(0)
    for _ in range(100):
        x = space.sample()
        assert x.shape == (4,)
        assert np.all(x >= -2.0) and np.all(x <= 3.0)


def test_box_vector_bounds():
    low = np.array([-1.0, 0.0])
    high = np.array([1.0, 5.0])
    space = Box(low, high)
    assert space.shape == (2,)
    assert np.array_equal(space.low, low.astype(np.float32))
    assert np.array_equal(space.high, high.astype(np.float32))


def test_is_discrete_and_flatdim():
    assert is_discrete(Discrete(4))
    assert not is_discrete(Box(-1, 1, shape=(2,)))
    assert flatdim(Discrete(4)) == 4
    assert flatdim(Box(-1, 1, shape=(3,))) == 3


def test_is_discrete_on_gymnasium_spaces():
    gym = pytest.importorskip("gymnasium")
    assert is_discrete(gym.spaces.Discrete(3))
    assert not is_discrete(gym.spaces.Box(-1, 1, shape=(2,)))
