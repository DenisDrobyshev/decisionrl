import pytest

from reinforce.cli import _coerce, _parse_overrides, build_parser, main


def test_coerce_types():
    assert _coerce("10") == 10 and isinstance(_coerce("10"), int)
    assert _coerce("1e-3") == 1e-3 and isinstance(_coerce("1e-3"), float)
    assert _coerce("true") is True and _coerce("False") is False
    assert _coerce("none") is None
    assert _coerce("hello") == "hello"


def test_parse_overrides():
    out = _parse_overrides(["n_steps=2048", "lr=3e-4", "double_q=true"])
    assert out == {"n_steps": 2048, "lr": 3e-4, "double_q": True}


def test_parser_builds():
    parser = build_parser()
    args = parser.parse_args(["train", "ppo", "CartPole", "--steps", "100"])
    assert args.algo == "ppo" and args.env == "CartPole" and args.steps == 100


def test_cli_list_runs(capsys):
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "ppo" in out and "CartPole" in out


def test_cli_train_eval_roundtrip(tmp_path):
    path = str(tmp_path / "q.pkl")
    assert main(["train", "qlearning", "GridWorld", "--steps", "3000", "--eval-episodes", "2",
                 "--save", path]) == 0
    assert main(["eval", "qlearning", "--env", "GridWorld", "--load", path, "--episodes", "2"]) == 0


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "reinforce" in capsys.readouterr().out


def test_cli_train_with_progress(tmp_path):
    assert main(["train", "qlearning", "GridWorld", "--steps", "1000",
                 "--eval-episodes", "2", "--progress"]) == 0


def test_cli_play(tmp_path):
    path = str(tmp_path / "q.pkl")
    assert main(["train", "qlearning", "GridWorld", "--steps", "2000", "--eval-episodes", "1",
                 "--save", path]) == 0
    assert main(["play", "qlearning", "--env", "GridWorld", "--load", path, "--episodes", "2"]) == 0
