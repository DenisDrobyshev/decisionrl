# reinforce

**A dependency-light, correctness-first reinforcement learning foundation.**

Readable like CleanRL, composable like Stable-Baselines3, and batteries-included
so it runs the moment you `pip install` it.

## Highlights

- **20+ algorithms** — tabular (Q-Learning, SARSA, Expected SARSA), model-based
  (Dyna-Q, **MBPO**), value-based (DQN, C51, QR-DQN, **Rainbow**), policy gradient
  / actor-critic (REINFORCE, A2C, PPO, **IMPALA**, Recurrent PPO, discrete SAC),
  continuous control (DDPG, TD3, SAC — with optional PER + n-step), offline
  (TD3+BC, IQL, CQL) — plus multi-agent PPO (self-play / IPPO).
- **Correctness-first** — proper `terminated`/`truncated` bootstrapping, GAE,
  target-policy smoothing, automatic entropy tuning, orthogonal init.
- **Dependency-light** — only NumPy + PyTorch in the core; Gymnasium optional.
- **Batteries included** — built-in environments (classic control + applied),
  image observations (CNN), vectorized envs (sync & multiprocessing), a CLI and
  a tuned-hyperparameter registry.
- **One API for everything** — every agent has `predict` / `learn` / `save` /
  `load`.

![Agents learning applied tasks](assets/learning_curves.png)

## Install

```bash
pip install git+https://github.com/DenisDrobyshev/reinforce.git
# with Gymnasium environments:
pip install "reinforce[gym] @ git+https://github.com/DenisDrobyshev/reinforce.git"
```

See [Getting started](getting-started.md) for a first training run.
