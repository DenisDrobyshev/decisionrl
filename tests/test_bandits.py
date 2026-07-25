"""Tests for contextual bandit algorithms and the linear contextual bandit env."""

import numpy as np

from decisionrl.bandits import (
    EpsilonGreedyBandit,
    LinearThompsonSampling,
    LinUCB,
    run_bandit,
)
from decisionrl.envs import ContextualBandit


def _mean_regret(agent_cls, seeds=3, horizon=2000, **kw):
    out = []
    for s in range(seeds):
        env = ContextualBandit(n_arms=6, n_features=8, horizon=horizon, seed=42)
        agent = agent_cls(6, 8, seed=s, **kw)
        out.append(run_bandit(agent, env, seed=s)["cumulative_regret"])
    return float(np.mean(out))


def _random_regret(horizon=2000, seed=0):
    env = ContextualBandit(n_arms=6, n_features=8, horizon=horizon, seed=42)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    total, done = 0.0, False
    while not done:
        _, _, term, trunc, info = env.step(int(rng.integers(6)))
        total += info["regret"]
        done = term or trunc
    return total


def test_env_reports_nonnegative_regret_and_valid_optimal_arm():
    env = ContextualBandit(n_arms=5, n_features=4, horizon=100, seed=1)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (4,)
    for _ in range(100):
        obs, reward, term, trunc, info = env.step(env.action_space.sample())
        assert info["regret"] >= -1e-9
        assert 0 <= info["optimal_arm"] < 5
        if trunc:
            break


def test_optimal_action_has_zero_regret():
    # Choosing the environment's reported optimal arm must incur no regret.
    env = ContextualBandit(n_arms=5, n_features=4, horizon=50, seed=2)
    obs, _ = env.reset(seed=0)
    for _ in range(50):
        best = int(np.argmax(env.theta @ obs))
        obs, _, term, trunc, info = env.step(best)
        assert abs(info["regret"]) < 1e-9
        if trunc:
            break


def test_linucb_and_thompson_beat_epsilon_greedy():
    linucb = _mean_regret(LinUCB, alpha=1.0)
    thompson = _mean_regret(LinearThompsonSampling, v=0.25)
    eps = _mean_regret(EpsilonGreedyBandit, epsilon=0.1)
    assert linucb < eps, f"LinUCB {linucb:.1f} should beat eps-greedy {eps:.1f}"
    assert thompson < eps, f"Thompson {thompson:.1f} should beat eps-greedy {eps:.1f}"


def test_regret_is_sublinear_versus_random():
    # A learner's cumulative regret should be a small fraction of a random policy's.
    linucb = _mean_regret(LinUCB, alpha=1.0)
    random = _random_regret()
    assert linucb < 0.1 * random, f"LinUCB {linucb:.1f} vs random {random:.1f}"


def test_update_accumulates_statistics():
    agent = LinUCB(3, 4, seed=0)
    x = np.ones(4)
    before = agent.A[1].copy()
    agent.update(x, arm=1, reward=2.0)
    assert np.allclose(agent.A[1], before + np.outer(x, x))
    assert np.allclose(agent.b[1], 2.0 * x)
    assert np.allclose(agent.A[0], np.eye(4))  # other arms untouched
