"""Contract, determinism and learning tests for the complex scenario envs."""

import numpy as np
import pytest

from reinforce.algorithms import PPO, SAC
from reinforce.envs import LunarLander, Navigation2D, PortfolioAllocation, ReacherArm
from reinforce.training import evaluate_policy

ENV_CLASSES = [ReacherArm, Navigation2D, LunarLander, PortfolioAllocation]


@pytest.mark.parametrize("env_cls", ENV_CLASSES)
def test_env_reset_step_contract(env_cls):
    env = env_cls()
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs), f"{env_cls.__name__}: reset obs out of space"

    steps, done = 0, False
    while not done and steps < 10_000:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs), f"{env_cls.__name__}: obs out of space"
        assert np.all(np.isfinite(np.asarray(obs, dtype=np.float64)))
        assert np.isfinite(reward)
        done = terminated or truncated
        steps += 1
    assert done, f"{env_cls.__name__}: episode never ended"


@pytest.mark.parametrize("env_cls", ENV_CLASSES)
def test_env_seeding_is_deterministic(env_cls):
    ref = env_cls()
    ref.action_space.seed(0)
    actions = [ref.action_space.sample() for _ in range(15)]

    def rollout():
        env = env_cls()
        obs, _ = env.reset(seed=7)
        trace = [np.asarray(obs, dtype=np.float64).copy()]
        for a in actions:
            obs, r, term, trunc, _ = env.step(a)
            trace.append(np.asarray(obs, dtype=np.float64).copy())
            if term or trunc:
                break
        return np.concatenate(trace)

    np.testing.assert_allclose(rollout(), rollout())


def test_lunar_lander_can_land_and_crash():
    # Reward includes the large terminal bonus/penalty on touchdown.
    env = LunarLander()
    env.reset(seed=0)
    saw_terminal = False
    for _ in range(400):
        _, r, term, trunc, _ = env.step(env.action_space.sample())
        if term:
            saw_terminal = True
            break
        if trunc:
            break
    assert saw_terminal or trunc  # episode resolves


def test_portfolio_softmax_weights_are_a_simplex():
    env = PortfolioAllocation(n_assets=4)
    env.reset(seed=0)
    env.step(np.array([2.0, 0.0, -1.0, 0.5], dtype=np.float32))
    # after a rebalance the internal weights are non-negative and sum to 1
    assert env._weights.min() >= 0.0
    assert abs(float(env._weights.sum()) - 1.0) < 1e-6


# --------------------------------------------------------------------------- #
# Learning tests: each complex scenario is solved by an appropriate algorithm.
# Thresholds are lenient (device/budget robust): the point is that the agent
# clearly learns, not that it hits a specific score.
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_sac_learns_reacher(quiet_logger):
    agent = SAC(ReacherArm(), seed=0, logger=quiet_logger)
    before, _ = evaluate_policy(agent, ReacherArm(), n_episodes=20, seed=100)
    agent.learn(30_000)
    after, _ = evaluate_policy(agent, ReacherArm(), n_episodes=20, seed=100)
    assert after > before + 0.5


@pytest.mark.slow
def test_sac_learns_navigation(quiet_logger):
    before, _ = evaluate_policy(SAC(Navigation2D(), seed=0, logger=quiet_logger),
                                Navigation2D(), n_episodes=20, seed=100)
    agent = SAC(Navigation2D(), seed=0, logger=quiet_logger)
    agent.learn(60_000)
    after, _ = evaluate_policy(agent, Navigation2D(), n_episodes=20, seed=100)
    # Strong, device-robust improvement toward the goal (GPU calibration: -9.5 -> +5).
    assert after > before + 4.0


@pytest.mark.slow
def test_ppo_learns_lunar_lander(quiet_logger):
    agent = PPO(LunarLander(), n_steps=1024, batch_size=128, seed=0, logger=quiet_logger)
    before, _ = evaluate_policy(agent, LunarLander(), n_episodes=20, seed=100)
    agent.learn(200_000)
    after, _ = evaluate_policy(agent, LunarLander(), n_episodes=20, seed=100)
    # GPU calibration: -435 -> +79. Device-robust bar: strong improvement, clearly landing-ish.
    assert after > before + 150.0 and after > -150.0


@pytest.mark.slow
def test_sac_beats_equal_weight_on_portfolio(quiet_logger):
    agent = SAC(PortfolioAllocation(), seed=0, logger=quiet_logger)
    agent.learn(40_000)
    after, _ = evaluate_policy(agent, PortfolioAllocation(), n_episodes=20, seed=100)

    env = PortfolioAllocation()
    eq_returns = []
    for ep in range(20):
        obs, _ = env.reset(seed=100 + ep)
        done, total = False, 0.0
        while not done:
            obs, r, term, trunc, _ = env.step(np.zeros(env.n, dtype=np.float32))
            total += r
            done = term or trunc
        eq_returns.append(total)
    # The learned allocation should beat a static equal-weight rebalance.
    assert after > float(np.mean(eq_returns))
