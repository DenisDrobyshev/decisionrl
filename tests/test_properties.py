"""Property-based tests (Hypothesis) for buffers, spaces, schedules and stats."""

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from decisionrl import make_agent, make_env
from decisionrl.algorithms import PPO, SAC
from decisionrl.buffers import ReplayBuffer
from decisionrl.core.spaces import Box, Discrete
from decisionrl.exploration import LinearSchedule
from decisionrl.utils import RunningMeanStd

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


# --------------------------------------------------------------------------- #
# Property-based tests over the algorithms themselves.
#
# Invariants that must hold for *any* observation and *any* reasonable
# hyperparameters: predictions stay inside the action space, and a save/load
# round-trip is a no-op on the (deterministic) policy. Fixtures are module- or
# session-scoped so the (relatively expensive) agents are built once and then
# probed with many Hypothesis-generated observations.
# --------------------------------------------------------------------------- #

ALGO_SETTINGS = settings(max_examples=30, deadline=None)

_CARTPOLE_OBS = tuple(make_env("CartPole").observation_space.shape)
_CARTPOLE_ACTION = make_env("CartPole").action_space
_POINTMASS_OBS = tuple(make_env("PointMass").observation_space.shape)
_POINTMASS_ACTION = make_env("PointMass").action_space

DISCRETE_ALGOS = ["dqn", "ppo", "a2c", "c51", "qrdqn", "sac_discrete"]
CONTINUOUS_ALGOS = ["ddpg", "td3", "sac"]


def _obs_strategy(shape):
    return hnp.arrays(np.float32, shape, elements=st.floats(-5.0, 5.0, width=32))


@pytest.fixture(scope="module")
def discrete_agents():
    return {name: make_agent(name, make_env("CartPole"), seed=0) for name in DISCRETE_ALGOS}


@pytest.fixture(scope="module")
def continuous_agents():
    return {name: make_agent(name, make_env("PointMass"), seed=0) for name in CONTINUOUS_ALGOS}


@ALGO_SETTINGS
@given(obs=_obs_strategy(_CARTPOLE_OBS))
def test_discrete_agents_predict_valid_actions(discrete_agents, obs):
    for name, agent in discrete_agents.items():
        for deterministic in (True, False):
            a = agent.predict(obs, deterministic=deterministic)
            assert isinstance(a, (int, np.integer)), (name, type(a))
            assert _CARTPOLE_ACTION.contains(int(a)), (name, a)


@ALGO_SETTINGS
@given(obs=_obs_strategy(_POINTMASS_OBS))
def test_continuous_agents_predict_within_bounds(continuous_agents, obs):
    for name, agent in continuous_agents.items():
        a = np.asarray(agent.predict(obs, deterministic=True), dtype=np.float32)
        assert a.shape == _POINTMASS_ACTION.low.shape, (name, a.shape)
        assert np.all(np.isfinite(a)), (name, a)
        assert np.all(a >= _POINTMASS_ACTION.low - 1e-4), (name, a)
        assert np.all(a <= _POINTMASS_ACTION.high + 1e-4), (name, a)


@pytest.fixture(scope="module")
def ppo_roundtrip(tmp_path_factory):
    agent = make_agent("ppo", make_env("CartPole"), seed=0)
    path = str(tmp_path_factory.mktemp("ppo") / "ppo.pt")
    agent.save(path)
    return agent, PPO.load(path, env=make_env("CartPole"))


@ALGO_SETTINGS
@given(obs=_obs_strategy(_CARTPOLE_OBS))
def test_ppo_save_load_predictions_match(ppo_roundtrip, obs):
    agent, loaded = ppo_roundtrip
    assert agent.predict(obs, deterministic=True) == loaded.predict(obs, deterministic=True)


@pytest.fixture(scope="module")
def sac_roundtrip(tmp_path_factory):
    agent = make_agent("sac", make_env("PointMass"), seed=0)
    path = str(tmp_path_factory.mktemp("sac") / "sac.pt")
    agent.save(path)
    return agent, SAC.load(path, env=make_env("PointMass"))


@ALGO_SETTINGS
@given(obs=_obs_strategy(_POINTMASS_OBS))
def test_sac_save_load_predictions_match(sac_roundtrip, obs):
    agent, loaded = sac_roundtrip
    a1 = np.asarray(agent.predict(obs, deterministic=True))
    a2 = np.asarray(loaded.predict(obs, deterministic=True))
    np.testing.assert_allclose(a1, a2, atol=1e-5)


@settings(max_examples=15, deadline=None)
@given(
    learning_rate=st.floats(1e-5, 1e-2),
    gamma=st.floats(0.80, 0.999),
    h1=st.integers(8, 64),
    h2=st.integers(8, 64),
)
def test_dqn_random_hyperparams_construct_and_predict(learning_rate, gamma, h1, h2):
    env = make_env("CartPole")
    agent = make_agent(
        "dqn", env, learning_rate=learning_rate, gamma=gamma, hidden_sizes=(h1, h2), seed=0
    )
    obs, _ = env.reset(seed=0)
    action = agent.predict(obs)
    assert env.action_space.contains(int(action))
