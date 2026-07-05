import os

from reinforce.algorithms import QLearning
from reinforce.envs import GridWorld
from reinforce.training import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    ProgressBarCallback,
)


def test_progress_bar_callback_runs(quiet_logger):
    # Should work whether or not tqdm is installed (graceful no-op otherwise).
    agent = QLearning(GridWorld(), seed=0, logger=quiet_logger)
    agent.learn(500, callback=ProgressBarCallback())
    assert agent.num_timesteps >= 500


def test_checkpoint_callback_saves(tmp_path, quiet_logger):
    agent = QLearning(GridWorld(), seed=0, logger=quiet_logger)
    cb = CheckpointCallback(save_freq=200, save_dir=str(tmp_path), name_prefix="q", verbose=0)
    agent.learn(500, callback=cb)
    saved = [f for f in os.listdir(tmp_path) if f.startswith("q_")]
    assert len(saved) >= 2


def test_eval_callback_saves_best(tmp_path, quiet_logger):
    best = str(tmp_path / "best.pkl")
    agent = QLearning(GridWorld(), learning_rate=0.3, seed=0, logger=quiet_logger)
    cb = EvalCallback(GridWorld(), eval_freq=500, n_eval_episodes=3,
                      best_model_save_path=best, verbose=0)
    agent.learn(3000, callback=cb)
    assert os.path.exists(best)
    assert cb.best_mean_reward > -float("inf")
    assert len(cb.evaluations) > 0


def test_callback_list_combines(tmp_path, quiet_logger):
    agent = QLearning(GridWorld(), seed=0, logger=quiet_logger)
    ckpt = CheckpointCallback(save_freq=300, save_dir=str(tmp_path), name_prefix="c", verbose=0)
    evalc = EvalCallback(GridWorld(), eval_freq=300, n_eval_episodes=2, verbose=0)
    agent.learn(600, callback=CallbackList([ckpt, evalc]))
    assert any(f.startswith("c_") for f in os.listdir(tmp_path))
    assert len(evalc.evaluations) > 0
