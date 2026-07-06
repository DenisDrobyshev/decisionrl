"""Tests for the multi-process distributed actor-learner.

CartPole is passed as the (picklable) env factory; actors run in separate
processes with the spawn start method.
"""

import pytest

from reinforce.distributed import DistributedActorLearner
from reinforce.envs import CartPole
from reinforce.training import evaluate_policy


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
