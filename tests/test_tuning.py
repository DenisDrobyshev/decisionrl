import pytest


def test_optuna_search_qlearning():
    pytest.importorskip("optuna")
    from reinforce.envs import GridWorld
    from reinforce.tuning import optuna_search

    space = {
        "learning_rate": ("float", 0.05, 0.5, "log"),
        "gamma": ("categorical", [0.95, 0.99]),
    }
    study = optuna_search(
        "qlearning",
        lambda: GridWorld(rows=3, cols=3),
        space,
        n_trials=3,
        train_steps=3000,
        eval_episodes=3,
        seed=0,
    )
    assert len(study.trials) == 3
    assert study.best_value > -1e9
    assert "learning_rate" in study.best_params
    assert study.best_params["gamma"] in (0.95, 0.99)
