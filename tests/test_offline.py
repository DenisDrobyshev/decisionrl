import numpy as np
import pytest

from reinforce.algorithms import TD3BC
from reinforce.data import TransitionDataset, collect_dataset
from reinforce.envs import PointMass
from reinforce.training import evaluate_policy


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


@pytest.mark.slow
def test_td3bc_learns_offline(quiet_logger):
    rng = np.random.default_rng(0)

    def behavior(o):
        return np.clip(-np.asarray(o) * 3 + 0.3 * rng.standard_normal(2), -1, 1).astype(np.float32)

    ds = collect_dataset(PointMass(), behavior, n_transitions=15_000, gamma=0.99, seed=0)
    agent = TD3BC(PointMass(), alpha=2.5, batch_size=256, seed=0, logger=quiet_logger)
    agent.learn_offline(ds, total_steps=8_000)

    learned = evaluate_policy(agent, PointMass(), n_episodes=20, seed=1)[0]
    random_ret = _random_return(PointMass)
    assert learned > random_ret + 20, f"TD3BC offline: learned={learned:.2f} vs random={random_ret:.2f}"
