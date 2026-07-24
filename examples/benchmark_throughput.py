"""Computational throughput benchmark: how fast does the code run, not how well it learns.

Reports three numbers per algorithm on the chosen device:

* env steps/sec        - pure environment stepping under a random policy (the Python
                         simulation ceiling, independent of any learning)
* train steps/sec      - full training loop (act, step, store, gradient update)
* predict calls/sec    - deterministic inference latency of the learned policy

These measure wall-clock throughput, which is separate from the learning-quality
numbers in docs/benchmarks.md. Small networks on small environments are dominated by
Python env stepping and per-call overhead, so a single CPU thread is often the fastest
configuration; a GPU pays off only once the networks or batches are large.

Run: python examples/benchmark_throughput.py [--device cpu|cuda] [--steps 5000] [--compile]
"""

from __future__ import annotations

import argparse
import time

import torch

from decisionrl.algorithms import DQN, PPO, SAC
from decisionrl.envs import CartPole, Pendulum
from decisionrl.utils import Logger, set_seed
from decisionrl.utils.torch_utils import maybe_compile

# (name, agent class, env factory, agent kwargs)
CASES = [
    ("PPO", PPO, CartPole, dict(n_steps=1024, batch_size=64, n_epochs=10)),
    ("DQN", DQN, CartPole, dict(buffer_size=50_000, learning_starts=1000)),
    ("SAC", SAC, Pendulum, dict(buffer_size=50_000, learning_starts=1000, batch_size=256)),
]


def env_steps_per_sec(env_fn, steps: int) -> float:
    env = env_fn()
    obs, _ = env.reset(seed=0)
    t0 = time.perf_counter()
    for _ in range(steps):
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            obs, _ = env.reset()
    return steps / (time.perf_counter() - t0)


def train_steps_per_sec(agent_cls, env_fn, steps: int, device: str, compile_policy: bool, **kw):
    set_seed(0)
    agent = agent_cls(env_fn(), seed=0, logger=Logger(verbose=0), device=device, **kw)
    if compile_policy:
        for attr in ("policy", "actor", "q_net", "q1"):
            mod = getattr(agent, attr, None)
            if isinstance(mod, torch.nn.Module):
                setattr(agent, attr, maybe_compile(mod, enabled=True))
    t0 = time.perf_counter()
    agent.learn(steps)
    train_sps = steps / (time.perf_counter() - t0)

    env = env_fn()
    obs, _ = env.reset(seed=1)
    n = 2000
    t0 = time.perf_counter()
    for _ in range(n):
        agent.predict(obs, deterministic=True)
    predict_ps = n / (time.perf_counter() - t0)
    return train_sps, predict_ps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--steps", type=int, default=5000, help="training steps per case")
    ap.add_argument("--compile", action="store_true", help="wrap policy nets with torch.compile")
    args = ap.parse_args()
    torch.set_num_threads(1)

    print(f"device={args.device}  steps={args.steps}  compile={args.compile}  "
          f"torch={torch.__version__}")
    print(f"\n{'algo':<6}{'env':<12}{'env steps/s':>14}{'train steps/s':>16}{'predict/s':>12}")
    print("-" * 60)
    for name, cls, env_fn, kw in CASES:
        env_sps = env_steps_per_sec(env_fn, args.steps)
        train_sps, predict_ps = train_steps_per_sec(
            cls, env_fn, args.steps, args.device, args.compile, **kw)
        print(f"{name:<6}{env_fn.__name__:<12}{env_sps:>14,.0f}"
              f"{train_sps:>16,.0f}{predict_ps:>12,.0f}")


if __name__ == "__main__":
    main()
