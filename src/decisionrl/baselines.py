"""Classic (non-learning) baselines for the applied environments.

These are the operations-research reference policies that a learned agent is
measured against — base-stock, best-fixed-price, bang-bang, admission thresholds,
greedy battery control — plus small helpers to evaluate and tune them. Keeping
them here (instead of copy-pasted in examples and tests) means the "RL vs the
classic tool" comparison is defined in exactly one place.

Every policy is a callable ``(env, obs) -> action``; :func:`rollout_return`
evaluates one, and :func:`best_of` grid-searches a family for the strongest
member (a fair, non-straw-man baseline).
"""

from __future__ import annotations

from typing import Callable, Iterable, Tuple

import numpy as np

from .core.env import Env

__all__ = [
    "rollout_return",
    "best_of",
    "base_stock",
    "analytic_base_stock_level",
    "best_base_stock",
    "fixed_action",
    "best_fixed_action",
    "admit_all",
    "value_threshold",
    "best_value_threshold",
    "bang_bang",
    "no_battery",
    "price_threshold_battery",
    "best_price_threshold_battery",
    "order_nothing",
    "supply_base_stock",
    "best_supply_base_stock",
]

Policy = Callable[[Env, np.ndarray], object]


# --------------------------------------------------------------------------- #
# evaluation helpers
# --------------------------------------------------------------------------- #
def rollout_return(env_fn: Callable[[], Env], policy: Policy,
                   episodes: int = 40, seed: int = 1) -> float:
    """Mean undiscounted return of a ``(env, obs) -> action`` policy."""
    returns = []
    for ep in range(episodes):
        env = env_fn()
        obs, _ = env.reset(seed=seed + ep)
        done, total = False, 0.0
        while not done:
            obs, reward, term, trunc, _ = env.step(policy(env, obs))
            total += reward
            done = term or trunc
        returns.append(total)
    return float(np.mean(returns))


def best_of(env_fn: Callable[[], Env], factory: Callable[[float], Policy],
            params: Iterable[float], episodes: int = 40, seed: int = 1) -> Tuple[float, float]:
    """Grid-search a policy family; return ``(best_param, best_return)``."""
    best_param, best_val = None, -np.inf
    for p in params:
        val = rollout_return(env_fn, factory(p), episodes=episodes, seed=seed)
        if val > best_val:
            best_param, best_val = p, val
    return float(best_param), float(best_val)  # type: ignore[arg-type]


def _poisson_ppf(q: float, mu: float, support: int = 1000) -> int:
    """Smallest k with P(D <= k) >= q for D ~ Poisson(mu) (NumPy-only, no SciPy)."""
    k = np.arange(support)
    logpmf = -mu + k * np.log(mu) - np.cumsum(np.log(np.maximum(k, 1)))
    cdf = np.cumsum(np.exp(logpmf))
    idx = int(np.searchsorted(cdf, q))
    return min(idx, support - 1)


# --------------------------------------------------------------------------- #
# inventory (single-echelon, discrete order)
# --------------------------------------------------------------------------- #
def base_stock(level: float) -> Policy:
    """Order up to ``level`` (obs[0] is inventory scaled by ``max_inventory``)."""
    def policy(env: Env, obs: np.ndarray) -> int:
        inv = obs[0] * env.max_inventory
        return int(np.clip(round(level - inv), 0, env.max_order))
    return policy


def analytic_base_stock_level(env: Env) -> int:
    """Newsvendor critical-fractile order-up-to level for ``InventoryManagement``.

    ``S* = Poisson_ppf(Cu / (Cu + Co), mean)`` with underage cost
    ``Cu = price - unit_cost + stockout_penalty`` and overage cost
    ``Co = holding_cost`` — the textbook base-stock formula. (Ordering cost is not
    an overage cost here: leftover stock is carried to the next period, not scrapped.)
    """
    cu = env.price - env.unit_cost + env.stockout_penalty
    co = env.holding_cost
    critical = cu / (cu + co)
    level = _poisson_ppf(critical, env.demand_mean)
    return int(np.clip(level, 0, env.max_inventory))


def best_base_stock(env_fn: Callable[[], Env], s_range: Iterable[float] = None,
                    episodes: int = 40, seed: int = 1) -> Tuple[float, float]:
    """Exhaustive best base-stock level. With the default full integer range this is
    the *exact* optimum within the base-stock family (a 1-D exhaustive search)."""
    if s_range is None:
        s_range = range(0, env_fn().max_inventory + 1)
    return best_of(env_fn, base_stock, s_range, episodes=episodes, seed=seed)


# --------------------------------------------------------------------------- #
# dynamic pricing (discrete price index)
# --------------------------------------------------------------------------- #
def fixed_action(action: int) -> Policy:
    """Always take the same discrete action (e.g. a fixed list price)."""
    def policy(env: Env, obs: np.ndarray) -> int:
        return int(action)
    return policy


def best_fixed_action(env_fn: Callable[[], Env], episodes: int = 40,
                      seed: int = 1) -> Tuple[float, float]:
    n = env_fn().action_space.n
    return best_of(env_fn, lambda a: fixed_action(int(a)), range(n), episodes=episodes, seed=seed)


# --------------------------------------------------------------------------- #
# queue admission control (obs = [queue/buffer, incoming_value])
# --------------------------------------------------------------------------- #
def admit_all() -> Policy:
    return lambda env, obs: 1


def value_threshold(theta: float) -> Policy:
    """Admit the job only if its value clears a fixed threshold ``theta``."""
    def policy(env: Env, obs: np.ndarray) -> int:
        return int(obs[1] >= theta)
    return policy


def best_value_threshold(env_fn: Callable[[], Env], episodes: int = 40,
                         seed: int = 1) -> Tuple[float, float]:
    return best_of(env_fn, value_threshold, np.linspace(0.0, 1.0, 21),
                   episodes=episodes, seed=seed)


# --------------------------------------------------------------------------- #
# thermostat (obs[0] = temperature error)
# --------------------------------------------------------------------------- #
def bang_bang() -> Policy:
    return lambda env, obs: np.array([1.0 if obs[0] < 0 else -1.0], dtype=np.float32)


# --------------------------------------------------------------------------- #
# energy microgrid (obs = [soc, price, gen, load, sin_t, cos_t])
# --------------------------------------------------------------------------- #
def no_battery() -> Policy:
    return lambda env, obs: np.array([0.0], dtype=np.float32)


def price_threshold_battery(low: float, high: "float | None" = None) -> Policy:
    """Greedy arbitrage: charge when the (normalized) price is low, discharge when high."""
    hi = 1.0 - low if high is None else high

    def policy(env: Env, obs: np.ndarray) -> np.ndarray:
        price = obs[1]
        if price < low:
            return np.array([1.0], dtype=np.float32)   # charge cheap energy
        if price > hi:
            return np.array([-1.0], dtype=np.float32)  # discharge into the peak
        return np.array([0.0], dtype=np.float32)
    return policy


def best_price_threshold_battery(env_fn: Callable[[], Env], episodes: int = 40,
                                 seed: int = 1) -> Tuple[float, float]:
    return best_of(env_fn, lambda low: price_threshold_battery(float(low)),
                   np.linspace(0.2, 0.5, 7), episodes=episodes, seed=seed)


# --------------------------------------------------------------------------- #
# supply chain (obs = [retail_inv, wh_inv, retail_pipe, wh_pipe, last_demand] / scale)
# --------------------------------------------------------------------------- #
def order_nothing() -> Policy:
    return lambda env, obs: np.zeros(2, dtype=np.float32)


def supply_base_stock(level: float) -> Policy:
    """Per-echelon base-stock: bring each echelon's (on-hand + pipeline) up to ``level``."""
    def policy(env: Env, obs: np.ndarray) -> np.ndarray:
        scale = getattr(env, "_scale", 40.0)
        ri, wi, rp, wp, _ = np.asarray(obs) * scale
        return np.array([max(0.0, level - (ri + rp)) / env.max_order,
                         max(0.0, level - (wi + wp)) / env.max_order], dtype=np.float32)
    return policy


def best_supply_base_stock(env_fn: Callable[[], Env], s_range: Iterable[float] = range(8, 22),
                           episodes: int = 40, seed: int = 1) -> Tuple[float, float]:
    return best_of(env_fn, supply_base_stock, s_range, episodes=episodes, seed=seed)
