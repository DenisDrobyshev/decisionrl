"""Tests for intrinsic-motivation exploration (RND, ICM) and CuriosityWrapper."""

import numpy as np

from reinforce.algorithms import DQN
from reinforce.envs import CartPole
from reinforce.exploration import ICM, RND, CuriosityWrapper


def test_rnd_novelty_decreases_for_seen_states():
    rnd = RND(obs_dim=4, feature_dim=32)
    seen = np.ones(4, dtype=np.float32)
    unseen = np.full(4, -3.0, dtype=np.float32)

    before = rnd.intrinsic_reward(None, None, seen)
    for _ in range(200):
        rnd.update(None, None, seen)
    after = rnd.intrinsic_reward(None, None, seen)

    # Predictor learns the seen state -> its novelty drops sharply and falls
    # below the novelty of a never-seen state.
    assert after < before
    assert after < rnd.intrinsic_reward(None, None, unseen)


def test_icm_update_is_finite_and_trains():
    env = CartPole()
    icm = ICM(obs_dim=4, action_space=env.action_space, feature_dim=32)
    s = np.ones(4, dtype=np.float32)
    s2 = np.full(4, 0.5, dtype=np.float32)
    loss = icm.update(s, 1, s2)
    reward = icm.intrinsic_reward(s, 1, s2)
    assert np.isfinite(loss) and np.isfinite(reward) and reward >= 0.0


def test_curiosity_wrapper_augments_reward_and_preserves_components():
    env = CuriosityWrapper(CartPole(), RND(obs_dim=4), intrinsic_coef=0.5, normalize=True)
    obs, _ = env.reset(seed=0)
    next_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

    assert "extrinsic_reward" in info and "intrinsic_reward" in info
    assert info["extrinsic_reward"] == 1.0  # CartPole gives +1 per step
    assert info["intrinsic_reward"] >= 0.0
    # total = extrinsic + coef * normalized_intrinsic
    assert reward >= info["extrinsic_reward"]
    # observation is passed through unchanged
    assert env.observation_space.contains(np.asarray(next_obs, dtype=np.float32))


def test_agent_trains_on_curiosity_wrapped_env(quiet_logger):
    # Any agent works transparently on a curiosity-wrapped env.
    env = CuriosityWrapper(CartPole(), ICM(obs_dim=4, action_space=CartPole().action_space))
    agent = DQN(env, learning_starts=50, batch_size=16, buffer_size=1000, seed=0, logger=quiet_logger)
    agent.learn(300)
    assert agent.num_timesteps >= 300
    action = agent.predict(env.reset(seed=0)[0])
    assert env.action_space.contains(int(action))
