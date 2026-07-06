"""Property-based tests (Hypothesis) for buffers, spaces, schedules and stats."""

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from reinforce.buffers import ReplayBuffer
from reinforce.core.spaces import Box, Discrete
from reinforce.exploration import LinearSchedule
from reinforce.utils import RunningMeanStd

SETTINGS = settings(max_examples=50, deadline=None)


@SETTINGS
@given(n=st.integers(1, 20), start=st.integers(-5, 5))
def test_discrete_sample_always_valid(n, start):
    space = Discrete(n, start=start)
    space.seed(0)
    for _ in range(20):
        a = space.sample()
        assert start <= a < start + n
        assert space.contains(a)


@SETTINGS
@given(
    lo=st.floats(-100, 100),
    span=st.floats(0.1, 100),
    dim=st.integers(1, 5),
)
def test_box_sample_within_bounds(lo, span, dim):
    low = np.full(dim, lo, dtype=np.float32)
    high = np.full(dim, lo + span, dtype=np.float32)
    space = Box(low, high)
    space.seed(0)
    for _ in range(20):
        x = space.sample()
        assert np.all(x >= low - 1e-4) and np.all(x <= high + 1e-4)
        assert space.contains(x)


@SETTINGS
@given(n_add=st.integers(1, 60), cap=st.integers(1, 25), dim=st.integers(1, 4))
def test_replay_storage_integrity(n_add, cap, dim):
    buf = ReplayBuffer(cap, Box(-1e9, 1e9, shape=(dim,)), Discrete(3), seed=0)
    for k in range(n_add):
        obs = np.full(dim, k, np.float32)
        buf.add(obs, k % 3, float(k), obs + 1.0, False)
    assert len(buf) == min(n_add, cap)
    batch = buf.sample(8)
    assert batch.obs.shape == (8, dim) and batch.next_obs.shape == (8, dim)
    # invariant preserved through ring storage: next_obs == obs + 1 everywhere
    assert torch.allclose(batch.next_obs, batch.obs + 1.0)
    assert (batch.actions >= 0).all() and (batch.actions < 3).all()


@SETTINGS
@given(
    rewards=st.lists(st.floats(-10, 10), min_size=1, max_size=5),
    gamma=st.floats(0.0, 0.999),
)
def test_nstep_return_is_discounted_sum(rewards, gamma):
    n = len(rewards)
    buf = ReplayBuffer(50, Box(-1, 1, shape=(1,)), Discrete(2), n_step=n, gamma=gamma)
    for i, r in enumerate(rewards):
        buf.add(np.zeros(1, np.float32), 0, r, np.zeros(1, np.float32), False, episode_end=(i == n - 1))
    expected = sum((gamma ** k) * rewards[k] for k in range(n))
    assert buf.rewards[0] == pytest.approx(expected, abs=1e-3)


@SETTINGS
@given(
    start=st.floats(0.0, 1.0),
    end=st.floats(0.0, 1.0),
    duration=st.integers(1, 10_000),
    step=st.integers(0, 50_000),
)
def test_linear_schedule_bounds_and_clamp(start, end, duration, step):
    sched = LinearSchedule(start, end, duration)
    v = sched(step)
    assert min(start, end) - 1e-6 <= v <= max(start, end) + 1e-6
    assert sched(duration) == pytest.approx(end)
    assert sched(duration + 1000) == pytest.approx(end)


@SETTINGS
@given(
    data=st.lists(st.floats(-100, 100), min_size=8, max_size=400),
)
def test_running_mean_std_matches_numpy(data):
    arr = np.array(data, dtype=np.float64).reshape(-1, 1)
    rms = RunningMeanStd(shape=(1,))
    for i in range(0, len(arr), 16):
        rms.update(arr[i : i + 16])
    # rtol/atol account for the small bias from the epsilon-count initialization
    np.testing.assert_allclose(rms.mean, arr.mean(axis=0), rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(rms.var, arr.var(axis=0), rtol=1e-2, atol=1e-2)
