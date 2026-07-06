"""Tests for the (experimental) Dreamer latent world-model agent.

We assert the world model actually learns the dynamics (its loss becomes small)
and that prediction/imagination run; we do not assert policy performance, since
this compact imagination-gradient variant is not tuned to beat the model-free
agents (see the class docstring).
"""

import numpy as np

from reinforce.algorithms import Dreamer
from reinforce.envs import PointMass
from reinforce.utils import HistoryLogger


def test_dreamer_constructs_and_predicts(quiet_logger):
    agent = Dreamer(PointMass(), latent_dim=16, horizon=3, learning_starts=10,
                    train_freq=10, model_updates=3, batch_size=16, seed=0, logger=quiet_logger)
    agent.learn(60)  # exercises world-model training + imagination behavior update
    obs, _ = PointMass().reset(seed=0)
    action = np.asarray(agent.predict(obs, deterministic=True))
    assert action.shape == (2,) and np.all(np.abs(action) <= 1.0 + 1e-5)


def test_dreamer_world_model_learns(quiet_logger):
    log = HistoryLogger()
    agent = Dreamer(PointMass(), latent_dim=32, horizon=5, learning_starts=200,
                    train_freq=50, model_updates=20, batch_size=128, seed=0, logger=log)
    agent.learn(2000)
    _, model_loss = log.curve("train/model_loss")
    assert model_loss, "no world-model loss was logged"
    assert model_loss[-1] < 0.05, f"world model did not learn the dynamics (loss={model_loss[-1]:.3f})"
