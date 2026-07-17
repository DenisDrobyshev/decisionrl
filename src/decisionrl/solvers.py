"""Exact dynamic-programming optima — the *provable* baseline where one exists.

For small, fully-observed, stationary problems the optimal policy can be computed
exactly by value iteration, with no learning. This is the strongest possible
baseline: a learned agent that *matches* it has recovered the true optimum. It is
also the honest boundary of the classical approach — the moment the problem stops
being a small stationary MDP (drifting demand, partial observability, coupled
continuous decisions) this tabular solver no longer applies, which is exactly where
reinforcement learning earns its place.

Pure NumPy, no external solver dependency.
"""

from __future__ import annotations

import numpy as np

from .core.env import Env

__all__ = ["value_iteration", "inventory_optimal_policy", "inventory_optimal_value"]


def value_iteration(P: np.ndarray, R: np.ndarray, gamma: float = 0.99,
                    tol: float = 1e-8, max_iter: int = 10_000) -> np.ndarray:
    """Optimal greedy policy for a finite MDP.

    ``P`` has shape ``(S, A, S)`` (transition probabilities) and ``R`` shape
    ``(S, A)`` (expected immediate reward). Returns the greedy policy as an
    ``(S,)`` array of action indices.
    """
    S, A, _ = P.shape
    V = np.zeros(S)
    for _ in range(max_iter):
        Q = R + gamma * np.einsum("sat,t->sa", P, V)
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    Q = R + gamma * np.einsum("sat,t->sa", P, V)
    return Q.argmax(axis=1)


def _inventory_mdp(env: Env):
    """Build ``(P, R)`` for the stationary inventory MDP from an env's parameters."""
    max_inv, max_order = env.max_inventory, env.max_order
    S, A = max_inv + 1, max_order + 1
    d_max = max_inv + int(6 * np.sqrt(env.demand_mean)) + 20
    d = np.arange(d_max + 1)
    logpmf = -env.demand_mean + d * np.log(env.demand_mean) - np.cumsum(np.log(np.maximum(d, 1)))
    pmf = np.exp(logpmf)
    pmf /= pmf.sum()  # normalize (lumps the negligible tail)

    P = np.zeros((S, A, S))
    R = np.zeros((S, A))
    for s in range(S):
        for a in range(A):
            inv_after = min(s + a, max_inv)
            sales = np.minimum(inv_after, d)
            lost = d - sales
            next_s = inv_after - sales
            reward = (env.price * sales - env.unit_cost * a
                      - env.holding_cost * next_s - env.stockout_penalty * lost)
            R[s, a] = float(pmf @ reward)
            np.add.at(P[s, a], next_s, pmf)
    return P, R


def inventory_optimal_policy(env: Env, gamma: float = 0.99):
    """Exact DP-optimal order-up policy for ``InventoryManagement`` as ``(env, obs) -> action``."""
    P, R = _inventory_mdp(env)
    pi = value_iteration(P, R, gamma=gamma)

    def policy(e: Env, obs: np.ndarray) -> int:
        s = int(round(obs[0] * e.max_inventory))
        return int(pi[min(max(s, 0), len(pi) - 1)])
    return policy


def inventory_optimal_value(env_fn, gamma: float = 0.99, episodes: int = 40, seed: int = 1) -> float:
    """Return of the exact DP-optimal inventory policy (same metric as everything else)."""
    from .baselines import rollout_return
    return rollout_return(env_fn, inventory_optimal_policy(env_fn(), gamma=gamma),
                          episodes=episodes, seed=seed)
