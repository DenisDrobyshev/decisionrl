import numpy as np
import pytest

from reinforce.core.spaces import is_discrete
from reinforce.envs import (
    Acrobot,
    CartPole,
    GridWorld,
    InventoryManagement,
    MountainCar,
    MountainCarContinuous,
    MultiArmedBandit,
    Pendulum,
    PointMass,
    Thermostat,
)

ENV_FACTORIES = [
    lambda: GridWorld(),
    lambda: GridWorld(one_hot=True),
    lambda: MultiArmedBandit(n_arms=5, seed=0),
    lambda: CartPole(),
    lambda: Pendulum(),
    lambda: PointMass(),
    lambda: InventoryManagement(),
    lambda: Thermostat(),
    lambda: MountainCar(),
    lambda: MountainCarContinuous(),
    lambda: Acrobot(),
]


@pytest.mark.parametrize("make_env", ENV_FACTORIES)
def test_reset_step_contract(make_env):
    env = make_env()
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs), f"reset obs not in space for {type(env).__name__}"
    assert isinstance(info, dict)

    action = env.action_space.sample()
    result = env.step(action)
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


@pytest.mark.parametrize("make_env", ENV_FACTORIES)
def test_reset_seed_reproducible(make_env):
    e1, e2 = make_env(), make_env()
    o1, _ = e1.reset(seed=42)
    o2, _ = e2.reset(seed=42)
    np.testing.assert_allclose(np.asarray(o1), np.asarray(o2))


def test_gridworld_reaches_goal_terminates():
    env = GridWorld(rows=2, cols=2, start=(0, 0), goal=(1, 1))
    env.reset(seed=0)
    # move right then down -> goal
    _, _, t1, _, _ = env.step(1)  # right
    _, r, t2, _, _ = env.step(2)  # down -> goal
    assert not t1 and t2
    assert r == pytest.approx(env.goal_reward)


def test_gridworld_walls_block():
    env = GridWorld(rows=3, cols=3, start=(0, 0), walls=[(0, 1)])
    env.reset(seed=0)
    obs, _, _, _, _ = env.step(1)  # try to move right into wall
    assert obs == 0  # stayed in place


def test_gridworld_truncates_at_max_steps():
    env = GridWorld(rows=5, cols=5, start=(0, 0), goal=(4, 4), max_steps=3)
    env.reset(seed=0)
    trunc = False
    for _ in range(3):
        _, _, term, trunc, _ = env.step(0)  # bump into top wall, never reach goal
    assert trunc and not term


def test_bandit_optimal_arm_info():
    env = MultiArmedBandit(means=[0.0, 5.0, 1.0], sigma=0.0, seed=0)
    env.reset(seed=0)
    assert env.optimal_arm == 1
    _, reward, terminated, _, info = env.step(1)
    assert terminated
    assert info["is_optimal"]
    assert reward == pytest.approx(5.0)


def test_cartpole_action_space():
    env = CartPole()
    assert is_discrete(env.action_space)
    assert env.action_space.n == 2
    assert env.observation_space.shape == (4,)


def test_pendulum_continuous_bounds():
    env = Pendulum()
    assert not is_discrete(env.action_space)
    obs, _ = env.reset(seed=0)
    # cos, sin in [-1, 1]
    assert -1.0 <= obs[0] <= 1.0 and -1.0 <= obs[1] <= 1.0


def test_pointmass_dense_reward_and_termination():
    env = PointMass(dim=2, max_steps=50, goal_radius=0.05)
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
    assert reward == pytest.approx(-info["distance"])
    assert not terminated  # started away from origin (very likely)


def test_inventory_truncates_and_bounds():
    env = InventoryManagement(max_inventory=20, max_order=10, horizon=10)
    env.reset(seed=0)
    truncated = False
    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert 0.0 <= obs[0] <= 1.0
        assert 0 <= env._inventory <= env.max_inventory
        assert {"demand", "sales", "lost_sales", "order"} <= set(info)
        assert not terminated
    assert truncated


def test_thermostat_truncates_and_power_clip():
    env = Thermostat(horizon=20)
    env.reset(seed=0)
    truncated = False
    for _ in range(20):
        obs, reward, terminated, truncated, info = env.step(np.array([5.0], np.float32))  # over-range
        assert -1.0 <= info["power"] <= 1.0  # action was clipped
        assert reward <= 0.0
        assert not terminated
    assert truncated


def test_gym_adapter():
    pytest.importorskip("gymnasium")
    from reinforce.envs import make_gym

    env = make_gym("CartPole-v1")
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs)
