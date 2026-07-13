import numpy as np
import pytest

from decisionrl.algorithms import CQL, IQL, TD3BC
from decisionrl.data import TransitionDataset, collect_dataset
from decisionrl.envs import PointMass
from decisionrl.training import evaluate_policy


def _random_return(env_fn, episodes=20, seed=100):
    rs = []
    for ep in range(episodes):
        env = env_fn()
        obs, _ = env.reset(seed=seed + ep)
        done, tot = False, 0.0
        while not done:
            obs, r, term, trunc, _ = env.step(env.action_space.sample())
            tot += r
            done = term or trunc
        rs.append(tot)
    return float(np.mean(rs))


def test_transition_dataset_sample():
    n, d = 50, 2
    ds = TransitionDataset(
        obs=np.zeros((n, d)), actions=np.zeros((n, d)), rewards=np.ones(n),
        next_obs=np.zeros((n, d)), dones=np.zeros(n), gamma=0.95, seed=0,
    )
    assert len(ds) == n
    batch = ds.sample(8)
    assert batch.obs.shape == (8, d)
    assert batch.actions.shape == (8, d)
    assert batch.discounts.shape == (8,)
    assert float(batch.discounts[0]) == pytest.approx(0.95)


def test_collect_dataset_length():
    ds = collect_dataset(PointMass(), lambda o: np.zeros(2, np.float32), n_transitions=200, seed=0)
    assert len(ds) == 200
    assert ds.obs.shape[1] == 2


def test_td3bc_online_learn_disabled(quiet_logger):
    agent = TD3BC(PointMass(), batch_size=8, seed=0, logger=quiet_logger)
    with pytest.raises(NotImplementedError):
        agent.learn(100)


def _behavior_dataset(n=15_000):
    rng = np.random.default_rng(0)

    def behavior(o):
        return np.clip(-np.asarray(o) * 3 + 0.3 * rng.standard_normal(2), -1, 1).astype(np.float32)

    return collect_dataset(PointMass(), behavior, n_transitions=n, gamma=0.99, seed=0)


def test_iql_online_learn_disabled(quiet_logger):
    agent = IQL(PointMass(), batch_size=8, seed=0, logger=quiet_logger)
    with pytest.raises(NotImplementedError):
        agent.learn(100)


def test_iql_predict_within_bounds(quiet_logger):
    agent = IQL(PointMass(), batch_size=8, seed=0, logger=quiet_logger)
    obs, _ = PointMass().reset(seed=0)
    action = np.asarray(agent.predict(obs, deterministic=True))
    assert action.shape == (2,)
    assert np.all(action >= -1.0 - 1e-5) and np.all(action <= 1.0 + 1e-5)


@pytest.mark.slow
def test_td3bc_learns_offline(quiet_logger):
    ds = _behavior_dataset()
    agent = TD3BC(PointMass(), alpha=2.5, batch_size=256, seed=0, logger=quiet_logger)
    agent.learn_offline(ds, total_steps=8_000)

    learned = evaluate_policy(agent, PointMass(), n_episodes=20, seed=1)[0]
    random_ret = _random_return(PointMass)
    assert learned > random_ret + 20, f"TD3BC offline: learned={learned:.2f} vs random={random_ret:.2f}"


@pytest.mark.slow
def test_iql_learns_offline(quiet_logger):
    ds = _behavior_dataset()
    agent = IQL(PointMass(), batch_size=256, expectile=0.7, beta=3.0, seed=0, logger=quiet_logger)
    agent.learn_offline(ds, total_steps=8_000)

    learned = evaluate_policy(agent, PointMass(), n_episodes=20, seed=1)[0]
    random_ret = _random_return(PointMass)
    assert learned > random_ret + 20, f"IQL offline: learned={learned:.2f} vs random={random_ret:.2f}"


def test_cql_online_learn_disabled(quiet_logger):
    agent = CQL(PointMass(), n_random=2, batch_size=8, seed=0, logger=quiet_logger)
    with pytest.raises(NotImplementedError):
        agent.learn(100)


def test_cql_predict_within_bounds(quiet_logger):
    agent = CQL(PointMass(), n_random=2, batch_size=8, seed=0, logger=quiet_logger)
    obs, _ = PointMass().reset(seed=0)
    action = np.asarray(agent.predict(obs, deterministic=True))
    assert action.shape == (2,)
    assert np.all(action >= -1.0 - 1e-5) and np.all(action <= 1.0 + 1e-5)


@pytest.mark.slow
def test_cql_learns_offline(quiet_logger):
    ds = _behavior_dataset()
    agent = CQL(PointMass(), cql_alpha=1.0, n_random=4, batch_size=128, seed=0, logger=quiet_logger)
    agent.learn_offline(ds, total_steps=2_500)

    learned = evaluate_policy(agent, PointMass(), n_episodes=20, seed=1)[0]
    random_ret = _random_return(PointMass)
    assert learned > random_ret + 20, f"CQL offline: learned={learned:.2f} vs random={random_ret:.2f}"
