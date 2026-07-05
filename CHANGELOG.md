# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **n-step returns** in `ReplayBuffer`/`PrioritizedReplayBuffer` (per-sample
  discount, correct across termination and truncation); `n_step` option on DQN
  and the continuous off-policy agents.
- **C51** distributional DQN (`CategoricalQNetwork` + categorical projection).
- **Offline RL**: `TD3BC` (TD3 + behavior cloning) with `learn_offline`, plus a
  `TransitionDataset` / `collect_dataset` data module.
- **CLI**: `reinforce train|eval|list` console script and `python -m reinforce`.
- **Registry & configs**: `make_agent` / `make_env`, per-(algo, env) tuned
  hyperparameters.
- **AsyncVectorEnv**: subprocess-based vectorized env (spawn) with the same API
  as `SyncVectorEnv`, for parallel data collection; `maybe_compile` torch.compile
  helper.
- `py.typed` marker, pre-commit config, PyPI publish workflow (trusted publishing).

## [0.1.0] - 2026-07-05

Initial release.

### Added
- **Core**: Gymnasium-compatible `Env`/`Wrapper`, dependency-free `Box`/`Discrete`
  spaces, `BaseAgent` with a unified `predict`/`learn`/`save`/`load` API.
- **Environments** (no external deps): classic control `GridWorld`,
  `MultiArmedBandit`, `CartPole`, `Pendulum`, `PointMass`; **applied**
  `InventoryManagement` (operations) and `Thermostat` (HVAC/energy); plus
  optional Gymnasium interop (`make_gym`, `GymAdapter`).
- **Algorithms**:
  - Tabular: `QLearning`, `SARSA`, `ExpectedSARSA`.
  - Value-based: `DQN` with Double, Dueling and Prioritized Replay options.
  - Policy gradient: `REINFORCE` (with baseline), `A2C`, `PPO`.
  - Continuous control: `DDPG`, `TD3`, `SAC` (automatic entropy tuning).
- **Buffers**: `ReplayBuffer`, `PrioritizedReplayBuffer` (sum-tree),
  `RolloutBuffer` with GAE and correct time-limit bootstrapping.
- **Networks**: `build_mlp` with orthogonal init, Q-networks (incl. dueling),
  categorical/Gaussian/squashed-Gaussian/deterministic policies, value & Q critics.
- **Exploration**: linear/exponential/constant schedules, Gaussian and
  Ornstein-Uhlenbeck action noise.
- **Wrappers**: `TimeLimit`, `NormalizeObservation`, `NormalizeReward`,
  `SyncVectorEnv`.
- **Utils**: `set_seed`, `Logger` (stdout/CSV/optional TensorBoard),
  `HistoryLogger` (in-memory learning curves), `RunningMeanStd`, polyak/hard
  updates, explained variance.
- **Training**: `evaluate_policy`, callbacks (`EvalCallback`,
  `StopOnRewardThreshold`).
- 92 tests (unit + learning) and GitHub Actions CI across Python 3.9–3.12.
