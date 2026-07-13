"""Hyperparameter search with Optuna.

Requires: pip install optuna
Run: python examples/tune_optuna.py
"""

from decisionrl.envs import CartPole
from decisionrl.tuning import optuna_search


def main() -> None:
    search_space = {
        "learning_rate": ("float", 1e-4, 1e-2, "log"),
        "n_steps": ("categorical", [512, 1024, 2048]),
        "ent_coef": ("float", 0.0, 0.02),
    }
    study = optuna_search(
        "ppo",
        CartPole,
        search_space,
        n_trials=15,
        train_steps=25_000,
        eval_episodes=10,
        seed=0,
    )
    print("\nBest value:", study.best_value)
    print("Best params:", study.best_params)


if __name__ == "__main__":
    main()
