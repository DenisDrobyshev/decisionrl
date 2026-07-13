"""Tests for the multi-process distributed actor-learner.

CartPole is passed as the (picklable) env factory; actors run in separate
processes with the spawn start method.
"""

import multiprocessing as mp

import pytest

from decisionrl.distributed import DistributedActorLearner, _recv_with_timeout
from decisionrl.envs import CartPole
from decisionrl.training import evaluate_policy


def test_recv_with_timeout_raises_when_actor_stalls():
    parent, child = mp.Pipe()
    # nothing is ever sent on `child` -> a stalled/crashed actor
    with pytest.raises(TimeoutError):
        _recv_with_timeout(parent, timeout=0.05)


def test_recv_with_timeout_returns_message():
    parent, child = mp.Pipe()
    child.send({"ok": 1})
    assert _recv_with_timeout(parent, timeout=1.0) == {"ok": 1}


def test_distributed_smoke(quiet_logger):
    learner = DistributedActorLearner(CartPole, num_actors=2, n_steps=16, seed=0, logger=quiet_logger)
    try:
        learner.learn(256)
        assert learner.num_timesteps >= 256
        obs, _ = CartPole().reset(seed=0)
        assert learner.predict(obs, deterministic=True) in (0, 1)
    finally:
        learner.close()


@pytest.mark.slow
def test_distributed_learns_cartpole(quiet_logger):
    learner = DistributedActorLearner(CartPole, num_actors=4, n_steps=32, ent_coef=0.01,
                                      learning_rate=3e-4, seed=0, logger=quiet_logger)
    try:
        learner.learn(40_000)
        mean = evaluate_policy(learner, CartPole(), n_episodes=20)[0]
        assert mean > 150, f"distributed learner failed on CartPole (mean_return={mean:.1f})"
    finally:
        learner.close()
