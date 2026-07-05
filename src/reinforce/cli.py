"""Command-line interface: ``reinforce train|eval|list``.

Examples
--------
    reinforce list
    reinforce train ppo CartPole --steps 50000 --save ppo.pt
    reinforce train dqn CartPole --set learning_rate=5e-4 --set buffer_size=100000
    reinforce eval ppo --env CartPole --load ppo.pt --episodes 20
    reinforce train ppo gym:LunarLander-v2 --steps 200000
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List

from .configs import get_hyperparams
from .registry import ALGORITHMS, list_algorithms, list_environments, make_agent, make_env
from .training import evaluate_policy
from .utils import set_seed


def _coerce(value: str) -> Any:
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def _parse_overrides(items: List[str]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        key, val = item.split("=", 1)
        overrides[key.strip()] = _coerce(val.strip())
    return overrides


def _cmd_list(_: argparse.Namespace) -> int:
    print("Algorithms:")
    for name in list_algorithms():
        print(f"  - {name}")
    print("\nEnvironments (built-in):")
    for name in list_environments():
        print(f"  - {name}")
    print("\nGymnasium: prefix an id with 'gym:' e.g. gym:LunarLander-v2")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    set_seed(args.seed)
    env = make_env(args.env)

    kwargs: Dict[str, Any] = {} if args.no_tuned else get_hyperparams(args.algo, args.env)
    kwargs.update(_parse_overrides(args.set))
    kwargs.setdefault("seed", args.seed)

    print(f"Training {args.algo} on {args.env} for {args.steps} steps")
    if kwargs:
        print(f"Hyperparameters: {kwargs}")
    agent = make_agent(args.algo, env, **kwargs)
    agent.learn(args.steps)

    eval_env = make_env(args.env)
    mean, std = evaluate_policy(agent, eval_env, n_episodes=args.eval_episodes)
    print(f"\nEvaluation over {args.eval_episodes} episodes: {mean:.2f} +/- {std:.2f}")

    if args.save:
        agent.save(args.save)
        print(f"Saved agent to {args.save}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    key = args.algo.lower()
    if key not in ALGORITHMS:
        raise SystemExit(f"unknown algorithm {args.algo!r}")
    env = make_env(args.env)
    agent = ALGORITHMS[key].load(args.load, env=env)
    mean, std = evaluate_policy(agent, env, n_episodes=args.episodes, deterministic=True)
    print(f"{args.algo} on {args.env}: {mean:.2f} +/- {std:.2f} over {args.episodes} episodes")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reinforce", description="Reinforcement learning CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="list available algorithms and environments")
    p_list.set_defaults(func=_cmd_list)

    p_train = sub.add_parser("train", help="train an agent")
    p_train.add_argument("algo", help="algorithm name (see 'reinforce list')")
    p_train.add_argument("env", help="environment name or gym:<id>")
    p_train.add_argument("--steps", type=int, default=50_000)
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--save", type=str, default=None, help="path to save the trained agent")
    p_train.add_argument("--eval-episodes", type=int, default=20)
    p_train.add_argument("--no-tuned", action="store_true", help="ignore tuned default hyperparameters")
    p_train.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                         help="override a hyperparameter (repeatable)")
    p_train.set_defaults(func=_cmd_train)

    p_eval = sub.add_parser("eval", help="evaluate a saved agent")
    p_eval.add_argument("algo", help="algorithm name")
    p_eval.add_argument("--env", required=True, help="environment name or gym:<id>")
    p_eval.add_argument("--load", required=True, help="path to a saved agent")
    p_eval.add_argument("--episodes", type=int, default=20)
    p_eval.set_defaults(func=_cmd_eval)

    return parser


def main(argv: List[str] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
