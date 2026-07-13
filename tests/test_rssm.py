"""Tests for the RSSM world model and the DreamerRSSM agent."""

import numpy as np
import pytest
import torch

from decisionrl.algorithms import RSSM, DreamerRSSM
from decisionrl.envs import Pendulum


def _pendulum_sequences(n_seq=40, length=20, seed=0):
    """Fixed-length Pendulum sequences from a random policy."""
    rng = np.random.default_rng(seed)
    env = Pendulum()
    seqs = []
    for _ in range(n_seq):
        obs, _ = env.reset(seed=int(rng.integers(1_000_000)))
        o, a, r = [], [], []
        for _ in range(length):
            act = env.action_space.sample()
            o.append(np.asarray(obs, np.float32))
            a.append(np.asarray(act, np.float32))
            obs, rew, term, trunc, _ = env.step(act)
            r.append(float(rew))
        seqs.append((np.stack(o), np.stack(a), np.asarray(r, np.float32)))
    return seqs


def _batch(seqs, bs, rng):
    idx = rng.integers(0, len(seqs), size=bs)
    o = torch.tensor(np.stack([seqs[i][0] for i in idx]))
    a = torch.tensor(np.stack([seqs[i][1] for i in idx]))
    r = torch.tensor(np.stack([seqs[i][2] for i in idx]))
    return o, a, r


def test_rssm_world_model_learns():
    seqs = _pendulum_sequences()
    obs_dim, act_dim = seqs[0][0].shape[1], seqs[0][1].shape[1]
    rssm = RSSM(obs_dim, act_dim, deter=64, stoch=16, hidden=64)
    opt = torch.optim.Adam(rssm.parameters(), lr=6e-4)
    rng = np.random.default_rng(0)

    o, a, r = _batch(seqs, 32, rng)
    initial = rssm.loss(o, a, r)[1]["recon"]
    info = initial
    for _ in range(400):
        o, a, r = _batch(seqs, 32, rng)
        loss, info = rssm.loss(o, a, r)
        opt.zero_grad()
        loss.backward()
        opt.step()
    # the world model learns to reconstruct observations
    assert info["recon"] < 0.5 * initial
    assert np.isfinite(info["kl"]) and info["kl"] >= 0.0


def test_dreamer_rssm_predicts_and_round_trips(tmp_path, quiet_logger):
    agent = DreamerRSSM(Pendulum(), seq_len=10, horizon=5, batch_size=16,
                        learning_starts=200, train_freq=100, seed=0, logger=quiet_logger)
    agent.learn(1500)
    agent.reset_states()
    obs, _ = Pendulum().reset(seed=0)
    action = np.asarray(agent.predict(obs, deterministic=True))
    assert action.shape == (1,)
    assert np.all(action >= Pendulum().action_space.low - 1e-4)
    assert np.all(action <= Pendulum().action_space.high + 1e-4)

    path = str(tmp_path / "dreamer.pt")
    agent.save(path)
    loaded = DreamerRSSM.load(path, env=Pendulum())
    assert loaded.seq_len == agent.seq_len


@pytest.mark.slow
def test_dreamer_rssm_world_model_improves(quiet_logger):
    agent = DreamerRSSM(Pendulum(), seq_len=15, horizon=8, batch_size=24,
                        learning_starts=1000, train_freq=200, model_updates=40, seed=0, logger=quiet_logger)
    agent.learn(20_000)
    assert len(agent.model_losses) > 1
    assert agent.model_losses[-1] < agent.model_losses[0]  # world model improves over training
