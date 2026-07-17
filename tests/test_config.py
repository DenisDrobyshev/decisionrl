"""Tests for the declarative config runner and the reproducibility manifest."""

import json

import pytest

from decisionrl import config, tracking
from decisionrl.cli import main as cli_main


def test_build_from_dict():
    agent, env = config.build({"env": "CartPole", "algo": "ppo", "seed": 0})
    assert env.action_space.n == 2
    obs, _ = env.reset(seed=0)
    assert agent.predict(obs) in (0, 1)


def test_build_supports_kwargs():
    agent, env = config.build({
        "env": {"name": "GridWorld", "rows": 4, "cols": 4},
        "algo": {"name": "dqn", "learning_rate": 5e-4},
        "seed": 1,
    })
    assert env.rows == 4 and agent.optimizer.param_groups[0]["lr"] == 5e-4


def test_build_requires_env_and_algo():
    with pytest.raises(KeyError):
        config.build({"env": "CartPole"})


def test_run_trains_and_evaluates():
    result = config.run({"env": "GridWorld", "algo": "qlearning", "seed": 0,
                         "total_steps": 2000, "eval_episodes": 5})
    assert result["total_steps"] == 2000
    assert "mean" in result and "std" in result


def test_run_writes_manifest(tmp_path):
    mpath = tmp_path / "manifest.json"
    config.run({"env": "GridWorld", "algo": "qlearning", "seed": 3,
                "total_steps": 1000, "eval_episodes": 3, "manifest": str(mpath)})
    manifest = json.loads(mpath.read_text())
    assert manifest["seed"] == 3
    assert manifest["config"]["algo"] == "qlearning"
    assert "torch" in manifest["versions"] and "decisionrl" in manifest["versions"]
    assert "total_steps" in manifest["metrics"]


def test_load_yaml_config(tmp_path):
    pytest.importorskip("yaml")
    p = tmp_path / "run.yaml"
    p.write_text("env: CartPole\nalgo:\n  name: ppo\n  n_steps: 128\nseed: 0\n")
    cfg = config.load_config(str(p))
    assert cfg["env"] == "CartPole" and cfg["algo"]["n_steps"] == 128


def test_load_rejects_unknown_format(tmp_path):
    p = tmp_path / "run.txt"
    p.write_text("nope")
    with pytest.raises(ValueError):
        config.load_config(str(p))


def test_run_manifest_fields():
    m = tracking.run_manifest({"env": "CartPole", "algo": "ppo"}, metrics={"mean": 1.0}, seed=7)
    assert m["seed"] == 7 and m["metrics"]["mean"] == 1.0
    assert "timestamp" in m and "platform" in m and "versions" in m


def test_cli_run_json_config(tmp_path, capsys):
    p = tmp_path / "run.json"
    p.write_text(json.dumps({"env": "GridWorld", "algo": "qlearning", "seed": 0,
                             "total_steps": 800, "eval_episodes": 3}))
    rc = cli_main(["run", str(p)])
    assert rc == 0
    assert "eval return" in capsys.readouterr().out
