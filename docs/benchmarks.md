# Benchmarks

Reproduced scores from [`examples/benchmark_scores.py`](https://github.com/DenisDrobyshev/decisionrl/blob/main/examples/benchmark_scores.py),
single seed (0), CPU. Return is the mean +/- std of the final deterministic policy over
evaluation episodes; **random** is a uniform-random policy on the same task for reference.

| Algorithm | Environment | Steps | Return (mean ± std) | Random | Time (s) |
|---|---|---:|---:|---:|---:|
| QLearning | GridWorld | 30,000 | 0.95 ± 0.00 | 0.49 | 0 |
| SARSA | GridWorld | 30,000 | 0.95 ± 0.00 | 0.44 | 1 |
| ExpectedSARSA | GridWorld | 30,000 | 0.95 ± 0.00 | 0.35 | 1 |
| DQN | CartPole | 40,000 | 103.20 ± 2.77 | 21.35 | 93 |
| C51 | CartPole | 40,000 | 500.00 ± 0.00 | 21.25 | 131 |
| QRDQN | CartPole | 40,000 | 222.40 ± 183.09 | 20.80 | 159 |
| SACDiscrete | CartPole | 15,000 | 389.40 ± 86.25 | 22.50 | 95 |
| PPO | CartPole | 40,000 | 500.00 ± 0.00 | 21.80 | 39 |
| A2C | CartPole | 40,000 | 500.00 ± 0.00 | 23.75 | 36 |
| REINFORCE | CartPole | 30,000 | 473.00 ± 44.85 | 22.10 | 11 |
| DDPG | Pendulum | 15,000 | -146.88 ± 89.75 | -1318.11 | 113 |
| TD3 | Pendulum | 15,000 | -156.05 ± 100.58 | -1314.10 | 98 |
| SAC | Pendulum | 15,000 | -148.98 ± 97.51 | -1311.03 | 160 |
| TD3BC | PointMass | 10,000 | -2.47 ± 1.24 | -36.92 | 70 |
| IQL | PointMass | 10,000 | -2.74 ± 1.25 | -36.84 | 114 |

_Total wall-clock: 1119s. GridWorld optimal ≈ 0.95; CartPole max = 500; Pendulum optimal ≈ -150 (higher is better); PointMass random ≈ -42._

## Comparison vs Stable-Baselines3 / CleanRL

The scores above are on the built-in environments. To compare `decisionrl` head-to-head
against established libraries on the *same* Gymnasium tasks, use
[`examples/benchmark_vs_baselines.py`](https://github.com/DenisDrobyshev/decisionrl/blob/main/examples/benchmark_vs_baselines.py).
It trains matched algorithms on the same env, over several seeds and an identical step
budget, and reports mean ± std return and wall-clock side by side (results saved to JSON).

```bash
pip install stable_baselines3          # the SB3 side is skipped if not installed
python examples/benchmark_vs_baselines.py --algos ppo --env CartPole-v1 \
    --seeds 5 --steps 100000
```

### Methodology

- **Same env, same budget, same seeds.** Both libraries train on the identical
  Gymnasium id for the identical `total_timesteps`, then evaluate the greedy policy
  over 20 episodes; each library's own `evaluate_policy` is used.
- **Library defaults.** Each side uses its own default hyperparameters (this measures
  the out-of-the-box experience, not a tuned bake-off). For a fair tuned comparison,
  pass matched hyperparameters to both.
- **Multiple seeds.** Report the mean and std of the per-seed evaluation returns.
- **CleanRL.** CleanRL ships single-file reference scripts rather than an installable
  package, so compare by running the corresponding script (e.g. `ppo.py`) with the same
  `--env-id`, `--total-timesteps` and `--seed`, and drop its reported return into the
  table below.

### Results

Actual head-to-head runs vs **Stable-Baselines3 2.9.0** (evaluation return over 20
episodes, mean ± std across 3 seeds, library-default hyperparameters, CPU):

| Algorithm | Environment | Steps | Seeds | decisionrl | SB3 2.9.0 |
|---|---|---:|---:|---:|---:|
| PPO | CartPole-v1 | 50,000 | 3 | **500.0 ± 0.0** | 500.0 ± 0.0 |
| DQN | CartPole-v1 | 50,000 | 3 | **327 ± 122** | 96 ± 57 |

- **PPO** reaches parity — both solve CartPole to 500/500 in the same wall-clock (~29 s/seed).
- **DQN**: `decisionrl` scores higher at this budget, but the number is honest about two
  things — it is **higher-variance** (per-seed 500 / 245 / 238) and **slower** per seed
  (~57 s vs SB3's ~22 s; SB3's data pipeline is more optimized). At 50k steps with default
  hyperparameters SB3's DQN under-performs on CartPole; more steps or tuning close the gap.

Reproduce with `python examples/benchmark_vs_baselines.py --algos ppo dqn --env CartPole-v1 --seeds 3 --steps 50000`.
CleanRL (single-file scripts, not a package) can be dropped into the same table by running
its `ppo.py` / `dqn.py` with matched `--env-id/--total-timesteps/--seed`.

Atari and MuJoCo tasks work the same way once their extras are installed
(`pip install "gymnasium[atari,accept-rom-license,mujoco]"`); `decisionrl` reaches them
via `make_env("gym:ALE/Breakout-v5")` and `make_atari`.
