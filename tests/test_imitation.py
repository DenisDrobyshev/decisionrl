"""Tests for imitation learning: BC, DAgger and GAIL."""

import numpy as np
import pytest

from reinforce.envs import CartPole
from reinforce.imitation import BC, GAIL, DAgger, GAILDiscriminator, collect_expert_dataset
from reinforce.training import evaluate_policy


def _expert(o):
    return 1 if (o[2] + 0.5 * o[3]) > 0 else 0  # a heuristic that balances CartPole


def test_bc_predicts_and_round_trips(tmp_path, quiet_logger):
    data = collect_expert_dataset(CartPole(), _expert, 800, seed=0)
    bc = BC(CartPole(), seed=0, logger=quiet_logger)
    bc.train(data, n_iters=100, batch_size=64)
    obs, _ = CartPole().reset(seed=0)
    assert CartPole().action_space.contains(int(bc.predict(obs)))

    path = str(tmp_path / "bc.pt")
    bc.save(path)
    loaded = BC.load(path, env=CartPole())
    for s in range(10):
        o, _ = CartPole().reset(seed=s)
        assert bc.predict(o) == loaded.predict(o)


def test_gail_discriminator_reward_is_finite():
    disc = GAILDiscriminator(4, CartPole().action_space)
    r = disc.reward(np.zeros(4, dtype=np.float32), 1)
    assert np.isfinite(r)


@pytest.mark.slow
def test_bc_imitates_expert(quiet_logger):
    data = collect_expert_dataset(CartPole(), _expert, 4000, seed=0)
    bc = BC(CartPole(), seed=0, logger=quiet_logger)
    bc.train(data, n_iters=1500, batch_size=64)
    mean_return, _ = evaluate_policy(bc, CartPole(), n_episodes=10, seed=100)
    assert mean_return > 200.0


@pytest.mark.slow
def test_dagger_imitates_expert(quiet_logger):
    dagger = DAgger(CartPole(), seed=0, logger=quiet_logger)
    dagger.learn_dagger(CartPole(), _expert, iterations=4, steps_per_iter=800, train_iters=500)
    mean_return, _ = evaluate_policy(dagger, CartPole(), n_episodes=10, seed=100)
    assert mean_return > 200.0


@pytest.mark.slow
def test_gail_imitates_expert(quiet_logger):
    data = collect_expert_dataset(CartPole(), _expert, 4000, seed=0)
    gail = GAIL(CartPole(), data, n_steps=1024, batch_size=64, n_epochs=4, seed=0, logger=quiet_logger)
    gail.learn(iterations=10, steps_per_iter=2048, disc_epochs=5)
    after, _ = evaluate_policy(gail, CartPole(), n_episodes=10, seed=100)
    # GAIL matches the expert from demonstrations alone (no env reward); random ~= 22.
    assert after > 200.0
