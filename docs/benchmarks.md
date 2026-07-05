# Benchmarks

Reproduced scores from [`examples/benchmark_scores.py`](https://github.com/DenisDrobyshev/reinforce/blob/main/examples/benchmark_scores.py),
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
