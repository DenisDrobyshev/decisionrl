import numpy as np
import pytest

from reinforce.algorithms import RecurrentPPO
from reinforce.envs import BernoulliBandit
from reinforce.meta import make_meta_bandit
from reinforce.utils.logger import Logger


def test_bernoulli_bandit_rewards_and_optimal():
    env = BernoulliBandit(probs=[0.1, 0.9, 0.5], seed=0)
    assert env.optimal_arm == 1
    _, info = env.reset()
    rewards = set()
    for _ in range(50):
        _, r, term, _, info = env.step(1)
        rewards.add(r)
        assert term  # bandit is a single-step MDP
    assert rewards <= {0.0, 1.0}
    assert info["optimal_arm"] == 1


def test_rl2_observation_augmentation():
    env = make_meta_bandit(n_arms=4, horizon=10, seed=0)
    # base obs (1) + one-hot action (4) + prev reward + prev done
    assert env.observation_space.shape == (1 + 4 + 2,)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (7,)
    assert np.allclose(obs, 0.0)  # no previous action/reward at trial start

    obs, r, _, _, _ = env.step(2)
    onehot = obs[1:5]
    assert onehot.argmax() == 2 and onehot.sum() == 1.0  # previous action encoded
    assert obs[5] == r  # previous reward encoded
    assert obs[6] == 1.0  # inner bandit episode terminated -> done flag set


def test_rl2_trial_length_and_truncation():
    env = make_meta_bandit(n_arms=3, horizon=8, seed=1)
    env.reset(seed=1)
    dones = 0
    for step in range(8):
        _, _, terminated, truncated, _ = env.step(step % 3)
        assert not terminated  # inner terminations are hidden; the trial is one episode
        dones += int(truncated)
    assert dones == 1  # truncates exactly once, at the horizon


def test_rl2_resamples_task_each_trial():
    env = make_meta_bandit(n_arms=5, horizon=5, seed=0)
    env.reset(seed=0)
    probs_a = env.env.probs.copy()
    env.reset()
    probs_b = env.env.probs.copy()
    assert not np.allclose(probs_a, probs_b)  # a fresh task per trial


def test_recurrent_ppo_trains_on_rl2env(quiet_logger):
    env = make_meta_bandit(n_arms=4, horizon=10, seed=0)
    agent = RecurrentPPO(env, n_steps=10, lstm_size=32, seed=0, logger=quiet_logger)
    agent.learn(200)
    assert agent.num_timesteps >= 200
    obs, _ = env.reset(seed=0)
    agent.reset_states()
    action = agent.predict(obs, deterministic=True)
    assert 0 <= action < 4


@pytest.mark.slow
def test_rl2_meta_learns_to_adapt():
    # A recurrent policy meta-trained across Bernoulli bandits should, at test
    # time and with no gradient steps, pull the best arm far more often than
    # chance -- online adaptation driven purely by its hidden state.
    from reinforce.wrappers import SyncVectorEnv

    n_arms, horizon = 5, 30

    def mk(i):
        return lambda: make_meta_bandit(n_arms=n_arms, horizon=horizon, seed=1000 + i)

    venv = SyncVectorEnv([mk(i) for i in range(32)])
    agent = RecurrentPPO(
        venv, n_steps=horizon, n_epochs=10, ent_coef=0.01, learning_rate=1e-3,
        lstm_size=128, gae_lambda=0.3, seed=0, logger=Logger(verbose=0),
    )
    agent.learn(500_000)

    rng = np.random.default_rng(123)
    optimal_pulls = 0
    total_pulls = 0
    for _ in range(400):
        env = make_meta_bandit(n_arms=n_arms, horizon=horizon, seed=int(rng.integers(2**31)))
        obs, _ = env.reset()
        agent.reset_states()
        for _ in range(horizon):
            action = agent.predict(obs, deterministic=False)
            obs, _, _, _, info = env.step(action)
            optimal_pulls += int(info["is_optimal"])
            total_pulls += 1
    optimal_rate = optimal_pulls / total_pulls
    # Random pulls the best of 5 arms 20% of the time; the meta-learner beats that
    # by a wide margin (~0.4 in practice).
    assert optimal_rate > 0.30, f"meta-learner did not adapt (optimal_rate={optimal_rate:.3f})"
