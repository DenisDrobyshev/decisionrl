"""Tuned default hyperparameters per (algorithm, environment).

A lightweight "RL Zoo": sensible starting points validated during development.
Used by the CLI as defaults; override any of them from the command line.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

__all__ = ["HYPERPARAMS", "get_hyperparams"]

HYPERPARAMS: Dict[Tuple[str, str], Dict[str, Any]] = {
    ("ppo", "CartPole"): dict(n_steps=1024, batch_size=64, n_epochs=10, learning_rate=3e-4),
    ("a2c", "CartPole"): dict(n_steps=16, learning_rate=7e-4),
    ("dqn", "CartPole"): dict(
        learning_rate=1e-3, buffer_size=50_000, learning_starts=1000,
        target_update_interval=500, exploration_fraction=0.2, double_q=True,
    ),
    ("sac", "Pendulum"): dict(learning_starts=1000, batch_size=256),
    ("td3", "Pendulum"): dict(learning_starts=1000, batch_size=256),
    ("ddpg", "Pendulum"): dict(learning_starts=1000, batch_size=256),
    ("ppo", "InventoryManagement"): dict(n_steps=1024, batch_size=64, n_epochs=10, ent_coef=0.01),
    ("sac", "Thermostat"): dict(learning_starts=1000, batch_size=256),
    ("qlearning", "GridWorld"): dict(learning_rate=0.2),
    ("sarsa", "GridWorld"): dict(learning_rate=0.2),
}


def get_hyperparams(algo: str, env: str) -> Dict[str, Any]:
    """Return a copy of the tuned defaults for ``(algo, env)`` (empty if none)."""
    return dict(HYPERPARAMS.get((algo.lower(), env), {}))
