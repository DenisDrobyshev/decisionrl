"""API-conformance tests for the applied (operational-decision) environments."""

import numpy as np
import pytest

from decisionrl.envs import (
    DynamicPricing,
    EnergyMicrogrid,
    NonstationaryInventory,
    QueueAdmissionControl,
    SupplyChain,
)
from decisionrl.registry import make_env

APPLIED = [DynamicPricing, QueueAdmissionControl, EnergyMicrogrid, SupplyChain,
           NonstationaryInventory]
APPLIED_NAMES = ["DynamicPricing", "QueueAdmissionControl", "EnergyMicrogrid",
                 "SupplyChain", "NonstationaryInventory"]


@pytest.mark.parametrize("cls", APPLIED)
def test_reset_returns_valid_observation(cls):
    env = cls()
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)


@pytest.mark.parametrize("cls", APPLIED)
def test_rollout_conforms_to_step_api(cls):
    env = cls()
    env.reset(seed=0)
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward)
        assert isinstance(info, dict)
        steps += 1
        assert steps <= env.horizon + 1  # must terminate by the horizon
    assert truncated or terminated


@pytest.mark.parametrize("cls", APPLIED)
def test_reset_seed_is_deterministic(cls):
    a = cls().reset(seed=123)[0]
    b = cls().reset(seed=123)[0]
    assert np.allclose(a, b)


@pytest.mark.parametrize("name", APPLIED_NAMES)
def test_registered_in_make_env(name):
    env = make_env(name)
    assert env.observation_space is not None and env.action_space is not None


# --- slow: the learned policy must beat the naive operational baseline --------

def _baseline_return(env_cls, policy, episodes=30, seed=1):
    rs = []
    for ep in range(episodes):
        env = env_cls()
        obs, _ = env.reset(seed=seed + ep)
        done, tot = False, 0.0
        while not done:
            obs, r, term, trunc, _ = env.step(policy(obs))
            tot += r
            done = term or trunc
        rs.append(tot)
    return float(np.mean(rs))


@pytest.mark.slow
def test_queue_admission_beats_admit_all(quiet_logger):
    from decisionrl.algorithms import PPO
    from decisionrl.training import evaluate_policy

    admit_all = _baseline_return(QueueAdmissionControl, lambda o: 1)
    agent = PPO(QueueAdmissionControl(), n_steps=512, batch_size=64, n_epochs=10,
                seed=0, logger=quiet_logger)
    agent.learn(30_000)
    learned = evaluate_policy(agent, QueueAdmissionControl(), n_episodes=30, seed=1)[0]
    assert learned > admit_all + 10, f"learned={learned:.1f} admit_all={admit_all:.1f}"


@pytest.mark.slow
def test_dynamic_pricing_beats_random(quiet_logger):
    from decisionrl.algorithms import PPO
    from decisionrl.training import evaluate_policy

    rng = np.random.default_rng(0)
    n = DynamicPricing().action_space.n
    random_price = _baseline_return(DynamicPricing, lambda o: int(rng.integers(n)))
    agent = PPO(DynamicPricing(), n_steps=1024, batch_size=64, n_epochs=10, ent_coef=0.01,
                seed=0, logger=quiet_logger)
    agent.learn(50_000)
    learned = evaluate_policy(agent, DynamicPricing(), n_episodes=40, seed=1)[0]
    assert learned > random_price + 4, f"learned={learned:.1f} random={random_price:.1f}"


@pytest.mark.slow
def test_energy_microgrid_beats_no_battery(quiet_logger):
    from decisionrl.algorithms import SAC
    from decisionrl.training import evaluate_policy

    no_battery = _baseline_return(EnergyMicrogrid, lambda o: np.array([0.0], np.float32))
    agent = SAC(EnergyMicrogrid(), learning_starts=1000, batch_size=256, seed=0, logger=quiet_logger)
    agent.learn(20_000)
    learned = evaluate_policy(agent, EnergyMicrogrid(), n_episodes=40, seed=1)[0]
    assert learned > no_battery + 2, f"learned={learned:.1f} no_battery={no_battery:.1f}"


@pytest.mark.slow
def test_supply_chain_beats_order_nothing(quiet_logger):
    from decisionrl.algorithms import SAC
    from decisionrl.training import evaluate_policy

    nothing = _baseline_return(SupplyChain, lambda o: np.zeros(2, np.float32))
    agent = SAC(SupplyChain(), learning_starts=1000, batch_size=256, seed=0, logger=quiet_logger)
    agent.learn(20_000)
    learned = evaluate_policy(agent, SupplyChain(), n_episodes=40, seed=1)[0]
    assert learned > nothing + 60, f"learned={learned:.1f} order_nothing={nothing:.1f}"


@pytest.mark.slow
def test_nonstationary_inventory_beats_best_fixed_base_stock(quiet_logger):
    # The whole point of this env: under drifting demand, no fixed base-stock is
    # right, so an adaptive learned policy should beat the *best* fixed one. DQN is
    # used because it is stable here across seeds (PPO collapses on some seeds).
    from decisionrl.algorithms import DQN
    from decisionrl.training import evaluate_policy

    def base_stock_return(S, episodes=30):
        rs = []
        for ep in range(episodes):
            env = NonstationaryInventory()
            obs, _ = env.reset(seed=ep)
            done, tot = False, 0.0
            while not done:
                order = int(np.clip(round(S - obs[0] * env.max_inventory), 0, env.max_order))
                obs, r, term, trunc, _ = env.step(order)
                tot += r
                done = term or trunc
            rs.append(tot)
        return float(np.mean(rs))

    best_fixed = max(base_stock_return(S) for S in range(8, 24))
    agent = DQN(NonstationaryInventory(), learning_rate=5e-4, buffer_size=50_000,
                learning_starts=1000, target_update_interval=500, seed=0, logger=quiet_logger)
    agent.learn(100_000)
    learned = evaluate_policy(agent, NonstationaryInventory(), n_episodes=40, seed=1)[0]
    assert learned > best_fixed + 15, f"learned={learned:.1f} best_fixed_base_stock={best_fixed:.1f}"
