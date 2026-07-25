"""Tests for rliable-style evaluation statistics."""

import numpy as np

from decisionrl import baselines as B
from decisionrl.algorithms import QLearning
from decisionrl.envs import GridWorld, NonstationaryInventory
from decisionrl.evaluation import (
    aggregate_metrics,
    bootstrap_ci,
    iqm,
    performance_profile,
    probability_of_improvement,
    run_seeds,
    stress_test,
)


def test_stress_test_reports_every_variant():
    variants = {
        "nominal": NonstationaryInventory,
        "demand_spike": lambda: NonstationaryInventory(demand_low=8.0, demand_high=24.0),
    }
    result = stress_test(B.base_stock(10), variants, episodes=8)
    assert set(result) == {"nominal", "demand_spike"}
    assert all(isinstance(v, float) for v in result.values())


def test_adaptive_policy_is_more_robust_to_demand_spike():
    # A fixed base-stock tuned to nominal demand cannot serve a demand spike; the adaptive
    # tracking policy reads recent demand and keeps up, so it degrades far less.
    level, _ = B.best_base_stock(NonstationaryInventory, seed=0)
    safety, _ = B.best_tracking_base_stock(NonstationaryInventory, seed=0)
    spike = {"spike": lambda: NonstationaryInventory(demand_low=8.0, demand_high=24.0)}
    fixed = stress_test(B.base_stock(level), spike, episodes=30)["spike"]
    tracking = stress_test(B.tracking_base_stock(safety), spike, episodes=30)["spike"]
    assert tracking > fixed, f"tracking {tracking:.1f} should beat fixed {fixed:.1f} under a spike"


def test_iqm_ignores_tails():
    # middle 50% of 1..8 is 3,4,5,6 -> mean 4.5; outliers do not shift it much
    assert iqm(np.arange(1, 9)) == 4.5
    assert iqm([0, 5, 5, 5, 5, 100]) == 5.0


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    scores = rng.normal(10.0, 2.0, size=50)
    point, lo, hi = bootstrap_ci(scores, aggregate=np.mean, reps=1000, seed=0)
    assert lo <= point <= hi
    assert lo < 10.0 < hi  # true mean inside the interval


def test_aggregate_metrics_reports_three_estimators():
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    metrics = aggregate_metrics(scores, reps=500, seed=0)
    assert set(metrics) == {"mean", "median", "iqm"}
    for point, lo, hi in metrics.values():
        assert lo <= point <= hi


def test_performance_profile_is_monotone_in_unit_range():
    scores = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    taus = np.linspace(-1, 5, 25)
    frac = performance_profile(scores, taus)
    assert np.all((frac >= 0) & (frac <= 1))
    assert np.all(np.diff(frac) <= 1e-9)  # non-increasing
    assert frac[0] == 1.0  # every run beats a very low threshold


def test_probability_of_improvement():
    better = np.array([10.0, 11.0, 12.0])
    worse = np.array([1.0, 2.0, 3.0])
    assert probability_of_improvement(better, worse) == 1.0
    assert probability_of_improvement(worse, better) == 0.0
    assert probability_of_improvement(better, better) == 0.5


def test_run_seeds_returns_one_score_per_seed(quiet_logger):
    scores = run_seeds(
        lambda env: QLearning(env, seed=0, logger=quiet_logger),
        lambda: GridWorld(rows=3, cols=3),
        seeds=[0, 1, 2],
        steps=1500,
        eval_episodes=5,
    )
    assert scores.shape == (3,)
    assert np.all(np.isfinite(scores))
