"""Tests for the exact dynamic-programming solvers."""

import numpy as np

from decisionrl import baselines as B
from decisionrl import solvers
from decisionrl.envs import InventoryManagement


def test_value_iteration_solves_tiny_mdp():
    # Two states, two actions. Action 1 in state 0 pays 1 and stays; everything else 0.
    # The optimal policy takes action 1 in state 0.
    P = np.zeros((2, 2, 2))
    P[0, 0, 0] = 1.0
    P[0, 1, 0] = 1.0
    P[1, 0, 1] = 1.0
    P[1, 1, 1] = 1.0
    R = np.array([[0.0, 1.0], [0.0, 0.0]])
    pi = solvers.value_iteration(P, R, gamma=0.9)
    assert pi[0] == 1


def test_inventory_dp_matches_best_base_stock():
    # For stationary inventory, base-stock is provably optimal — DP must recover the
    # same value as the exhaustive best base-stock (within evaluation noise).
    dp = solvers.inventory_optimal_value(InventoryManagement, episodes=60, seed=0)
    _, bs = B.best_base_stock(InventoryManagement, episodes=60, seed=0)
    assert abs(dp - bs) < 0.05 * abs(bs)


def test_inventory_dp_beats_random():
    dp = solvers.inventory_optimal_value(InventoryManagement, episodes=40, seed=0)
    random = B.rollout_return(InventoryManagement, B.base_stock(0), episodes=40, seed=0)
    assert dp > random
