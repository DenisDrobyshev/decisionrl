"""Correctness tests for the applied-environment *dynamics* and their baselines.

These check the mechanics/economics are right (not just the Gym API shape and not
learning behaviour): a stochastic reward matches its parameters, a regime process
switches at the configured rate, the analytic base-stock is genuinely near-optimal,
demand responds to price, and conservation/bounds hold. Fast, deterministic.
"""

import numpy as np

from decisionrl import baselines as B
from decisionrl.envs import (
    BernoulliBandit,
    DynamicPricing,
    EnergyMicrogrid,
    InventoryManagement,
    NonstationaryInventory,
    QueueAdmissionControl,
    SupplyChain,
)


def test_bernoulli_bandit_reward_matches_probabilities():
    env = BernoulliBandit(probs=[0.2, 0.8], seed=0)
    means = []
    for arm in (0, 1):
        rs = []
        for _ in range(4000):
            env.reset()
            rs.append(env.step(arm)[1])
        means.append(np.mean(rs))
    assert abs(means[0] - 0.2) < 0.03 and abs(means[1] - 0.8) < 0.03


def test_pricing_demand_is_monotonic_in_price():
    env = DynamicPricing()
    demand_means = [env._demand_mean(p) for p in env.prices]
    assert all(a > b for a, b in zip(demand_means, demand_means[1:]))  # strictly decreasing


def test_analytic_base_stock_is_near_optimal_on_stationary_inventory():
    # The newsvendor critical-fractile level should be within a few % of the best
    # grid-searched base-stock — i.e. the formula genuinely solves the stationary case.
    s_star = B.analytic_base_stock_level(InventoryManagement())
    analytic = B.rollout_return(InventoryManagement, B.base_stock(s_star), episodes=60, seed=0)
    _, best = B.best_base_stock(InventoryManagement, episodes=60, seed=0)  # exhaustive optimum
    random = B.rollout_return(InventoryManagement, B.base_stock(0), episodes=60, seed=0)
    assert analytic >= 0.97 * best, f"analytic {analytic:.1f} vs exhaustive best {best:.1f}"
    assert analytic > random


def test_nonstationary_regime_switches_at_configured_rate():
    env = NonstationaryInventory(switch_prob=0.1)
    env.reset(seed=0)
    regimes = []
    for _ in range(6000):
        _, _, _, trunc, info = env.step(env.action_space.sample())
        regimes.append(info["regime"])
        if trunc:
            env.reset()
    switches = sum(a != b for a, b in zip(regimes, regimes[1:]))
    rate = switches / len(regimes)
    assert abs(rate - 0.1) < 0.03, f"observed switch rate {rate:.3f}"


def test_nonstationary_ewma_tracks_regime():
    # obs[1] is the EWMA of demand; it should read higher in the high regime.
    env = NonstationaryInventory()
    obs, _ = env.reset(seed=1)
    high_ewma, low_ewma = [], []
    for _ in range(4000):
        obs, _, _, trunc, info = env.step(0)  # order nothing; only observe demand
        (high_ewma if info["regime"] == "high" else low_ewma).append(obs[1])
        if trunc:
            obs, _ = env.reset()
    assert np.mean(high_ewma) > np.mean(low_ewma) + 0.1


def test_energy_soc_stays_within_battery_bounds():
    env = EnergyMicrogrid()
    env.reset(seed=0)
    done = False
    while not done:
        _, _, term, trunc, info = env.step(env.action_space.sample())
        assert -1e-6 <= info["soc"] <= env.capacity + 1e-6
        done = term or trunc
    # Charging from empty must raise the state of charge.
    env.reset(seed=0)
    env._soc = 0.0
    env.step(np.array([1.0], np.float32))
    assert env._soc > 0.0


def test_supply_chain_inventories_nonnegative():
    env = SupplyChain()
    env.reset(seed=0)
    done = False
    while not done:
        _, _, term, trunc, info = env.step(env.action_space.sample())
        assert info["retail_inv"] >= -1e-6 and info["warehouse_inv"] >= -1e-6
        done = term or trunc


def test_queue_never_exceeds_buffer():
    env = QueueAdmissionControl(buffer_size=5)
    env.reset(seed=0)
    done = False
    while not done:
        _, _, term, trunc, info = env.step(1)  # always try to admit
        assert 0 <= info["queue"] <= 5
        done = term or trunc
