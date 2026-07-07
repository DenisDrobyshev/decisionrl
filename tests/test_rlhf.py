"""Tests for the RLHF pipeline: reward model, preferences and env wrapper."""

import numpy as np

from reinforce.envs import PointMass
from reinforce.rlhf import (
    RewardModel,
    RewardModelWrapper,
    collect_segments,
    synthetic_preferences,
    train_reward_model,
)


def _random_policy(env):
    return lambda o: env.action_space.sample()


def test_reward_model_learns_preferences_and_recovers_true_reward():
    env = PointMass()
    segments = collect_segments(env, _random_policy(env), n_segments=120, seg_len=25, seed=0)
    prefs = synthetic_preferences(segments, n_pairs=800, rational=True, seed=1)

    # PointMass reward is state-only (-distance), so a state reward model suffices.
    model = RewardModel(obs_dim=2, action_space=env.action_space, use_action=False)
    metrics = train_reward_model(model, prefs, n_iters=500, batch_size=32)
    assert metrics["accuracy"] > 0.85

    # The learned reward should correlate strongly with the true reward r(s)=-||s||.
    grid = np.random.default_rng(2).uniform(-1, 1, size=(400, 2)).astype(np.float32)
    true_r = -np.linalg.norm(grid, axis=1)
    learned = model.predict_rewards(grid)
    corr = float(np.corrcoef(true_r, learned)[0, 1])
    assert corr > 0.8


def test_reward_model_wrapper_reports_true_reward():
    env = PointMass()
    model = RewardModel(obs_dim=2, action_space=env.action_space, use_action=False)
    wrapped = RewardModelWrapper(PointMass(), model)

    wrapped.reset(seed=0)
    obs, learned_reward, terminated, truncated, info = wrapped.step(wrapped.action_space.sample())
    assert "true_reward" in info
    assert isinstance(learned_reward, float)
    # the true reward is the real environment reward (negative distance)
    assert info["true_reward"] <= 0.0


def test_synthetic_preferences_label_higher_return():
    env = PointMass()
    segments = collect_segments(env, _random_policy(env), n_segments=40, seg_len=20, seed=3)
    prefs = synthetic_preferences(segments, n_pairs=200, rational=True, seed=4)
    for a, b, label in prefs.pairs:
        if label == 1.0:
            assert a.true_return >= b.true_return
        else:
            assert b.true_return > a.true_return
