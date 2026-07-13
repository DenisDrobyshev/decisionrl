"""Hyperparameter search with Optuna (optional dependency).

Give a search space as ``{name: spec}`` where ``spec`` is one of::

    ("float", low, high)            # uniform
    ("float", low, high, "log")     # log-uniform
    ("int", low, high)              # integer uniform
    ("int", low, high, "log")       # integer log-uniform
    ("categorical", [choices])      # choose from a list

Example
-------
>>> from decisionrl.tuning import optuna_search           # doctest: +SKIP
>>> from decisionrl.envs import CartPole                   # doctest: +SKIP
>>> space = {"learning_rate": ("float", 1e-4, 1e-2, "log"),
...          "n_steps": ("categorical", [512, 1024, 2048])}
>>> study = optuna_search("ppo", CartPole, space, n_trials=20, train_steps=30_000)  # doctest: +SKIP
>>> study.best_params                                     # doctest: +SKIP
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

from .registry import make_agent
from .training import evaluate_policy
from .utils import Logger, set_seed

__all__ = ["optuna_search"]


def _suggest(trial, name: str, spec: Tuple):
    kind = spec[0]
    if kind == "float":
        log = len(spec) > 3 and spec[3] == "log"
        return trial.suggest_float(name, spec[1], spec[2], log=log)
    if kind == "int":
        log = len(spec) > 3 and spec[3] == "log"
        return trial.suggest_int(name, spec[1], spec[2], log=log)
    if kind == "categorical":
        return trial.suggest_categorical(name, spec[1])
    raise ValueError(f"unknown search-space spec for {name!r}: {spec!r}")


def optuna_search(
    algo: str,
    env_fn: Callable[[], Any],
    search_space: Dict[str, Tuple],
    n_trials: int = 20,
    train_steps: int = 50_000,
    eval_episodes: int = 10,
    seed: int = 0,
    direction: str = "maximize",
    **agent_kwargs,
):
    """Run an Optuna study that maximizes evaluation return over ``search_space``.

    Returns the completed ``optuna.Study`` (see ``study.best_params`` /
    ``study.best_value``). Requires ``pip install optuna``.
    """
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("optuna is required for optuna_search: pip install optuna") from exc

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {name: _suggest(trial, name, spec) for name, spec in search_space.items()}
        set_seed(seed)
        agent = make_agent(algo, env_fn(), seed=seed, logger=Logger(verbose=0), **params, **agent_kwargs)
        agent.learn(train_steps)
        mean, _ = evaluate_policy(agent, env_fn(), n_episodes=eval_episodes)
        return mean

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)
    return study
