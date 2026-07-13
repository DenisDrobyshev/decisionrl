"""Tests for Hindsight Experience Replay (BitFlipping env + HERDQN)."""

import numpy as np
import pytest
import torch

from decisionrl.algorithms import HERDQN, HERReplayBuffer
from decisionrl.envs import BitFlipping


def test_bitflipping_mechanics():
    env = BitFlipping(n_bits=4)
    obs, info = env.reset(seed=0)
    assert obs.shape == (8,)  # concat(state, goal)
    assert "achieved_goal" in info and "desired_goal" in info
    assert env.compute_reward(np.ones(4), np.ones(4)) == 0.0
    assert env.compute_reward(np.zeros(4), np.ones(4)) == -1.0

    before = env._state.copy()
    env.step(2)
    assert env._state[2] == 1.0 - before[2]  # the chosen bit flipped


def test_her_buffer_relabels_and_shapes():
    env = BitFlipping(n_bits=4)
    buf = HERReplayBuffer(100, n_bits=4, compute_reward=env.compute_reward, her_ratio=1.0, seed=0)
    rng = np.random.default_rng(0)
    states = [np.concatenate([rng.integers(0, 2, 4), rng.integers(0, 2, 4)]).astype(np.float32) for _ in range(5)]
    next_states = [np.concatenate([rng.integers(0, 2, 4), s[4:]]).astype(np.float32) for s in states]
    buf.store_episode(states, [0, 1, 2, 3, 0], next_states)

    obs, act, rew, next_obs, done = buf.sample(16, torch.device("cpu"))
    assert obs.shape == (16, 8) and next_obs.shape == (16, 8)
    assert set(np.unique(rew.numpy()).tolist()).issubset({0.0, -1.0})
    # done is 1 exactly when the relabelled reward is 0 (goal reached)
    assert torch.all((done == 1.0) == (rew == 0.0))


def test_herdqn_predicts_and_round_trips(tmp_path, quiet_logger):
    env = BitFlipping(n_bits=4)
    agent = HERDQN(env, gradient_steps=2, batch_size=32, seed=0, logger=quiet_logger)
    agent.learn(200)
    obs, _ = env.reset(seed=0)
    assert env.action_space.contains(int(agent.predict(obs)))

    path = str(tmp_path / "her.pt")
    agent.save(path)
    loaded = HERDQN.load(path, env=BitFlipping(n_bits=4))
    assert loaded.predict(obs) == agent.predict(obs)


@pytest.mark.slow
def test_herdqn_solves_bitflipping(quiet_logger):
    agent = HERDQN(BitFlipping(n_bits=6), gradient_steps=20, batch_size=128, seed=0, logger=quiet_logger)
    agent.learn(5000)
    # sparse-reward BitFlipping is unsolvable by vanilla DQN; HER solves it.
    assert agent.success_rate(n_episodes=50) > 0.8
