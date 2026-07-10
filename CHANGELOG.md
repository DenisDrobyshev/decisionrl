# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Evolutionary & swarm optimization** (`reinforce.evolution`): a unified
  ask/tell family of gradient-free optimizers — evolution strategies (`CEM`,
  `CMAES`, `DifferentialEvolution`, `GeneticAlgorithm`, `OpenAIES`, `ARS`,
  `SimulatedAnnealing`) and swarm intelligence (`PSO`, `FireflyAlgorithm`,
  `ArtificialBeeColony`, `GreyWolfOptimizer`, `BatAlgorithm`, `AntColonyTSP`) —
  benchmark functions, and a `NeuroevolutionAgent` that trains RL policies with
  any optimizer (no gradients; solves CartPole). Figures via
  `examples/evolution_demo.py`.
- **Policy serving** (`reinforce.serving`): `export_onnx` / `export_torchscript`
  freeze the deterministic policy (+ JSON metadata); `OnnxPolicy` runs inference
  with onnxruntime alone; `create_app` exposes a FastAPI `/predict` service, with
  a `deploy/Dockerfile`. New `serve` optional-dependency extra.
- **Renders for the complex scenarios**: `render_rgb()` on `ReacherArm`,
  `Navigation2D` (with lidar) and `LunarLander` for GIFs/figures.
- **Complex scenario environments** (self-contained, NumPy-only): `ReacherArm`
  (2-link torque-controlled reaching), `Navigation2D` (continuous maze with lidar
  sensors and hard exploration), `LunarLander` (2-D rigid-body rocket landing with
  shaped + terminal reward), and `PortfolioAllocation` (allocate across correlated
  momentum assets with transaction costs). All registered in `make_env`; trained
  end-to-end in `examples/complex_scenarios.py`.
- **RLHF pipeline** (`reinforce.rlhf`): learn a reward from *preferences* the way
  language models are aligned — `collect_segments`, `synthetic_preferences`
  (Bradley-Terry teacher), a `RewardModel` trained on the preference likelihood,
  and `RewardModelWrapper` to optimize any agent against the learned reward while
  keeping the true reward in `info` for evaluation.
- **GRPO** (`GRPO`): Group Relative Policy Optimization (DeepSeekMath) — the
  critic-free, group-normalized-advantage policy-optimization method behind modern
  LLM RLHF, with a PPO clipped objective and a KL penalty to a reference policy.
  Pairs directly with `reinforce.rlhf`.
- **Decision Transformer** (`DecisionTransformer`): offline RL as return-conditioned
  sequence modeling (causal GPT over `(return-to-go, state, action)` tokens), with
  a `TrajectoryDataset` / `collect_trajectories` data module and a faithful
  return-conditioned `evaluate`.
- **Curiosity / intrinsic motivation** (`reinforce.exploration`): `RND` (Random
  Network Distillation) and `ICM` (Intrinsic Curiosity Module), plus a
  `CuriosityWrapper` that adds a normalized novelty bonus to any environment so
  every agent gets exploration for sparse-reward tasks for free.
- **Distributed actors** (`DistributedActorLearner`): true multi-process
  IMPALA-style training — actor processes run env + local inference and stream
  trajectories to a central V-trace learner that broadcasts fresh weights.
- **Dreamer** (`Dreamer`, experimental): compact latent world model with an
  actor-critic trained in imagination via analytic gradients through the learned
  dynamics. The world model learns the dynamics; the policy-learning part is not
  tuned to be competitive (use `MBPO` for a robust model-based agent).
- **Property-based tests** (Hypothesis) for spaces, replay buffer storage/n-step,
  schedules and running statistics, and for the **algorithms themselves**:
  predictions stay inside the action space for arbitrary observations, and
  save/load round-trips are a no-op on the deterministic policy.
- **PyPI packaging**: distributed as `reinforce-rl` (imported as `reinforce`);
  release process documented in [`RELEASING.md`](RELEASING.md) via trusted
  publishing. README gains a hero banner (`docs/assets/banner.svg`).
- **MBPO** (`MBPO`): model-based policy optimization — an `EnsembleDynamics`
  model generates short synthetic rollouts to augment SAC training on a mix of
  real and model data.
- **IMPALA** (`IMPALA`): V-trace off-policy actor-critic for parallel-actor
  training (sync/async vector envs); corrects behaviour/target policy lag across
  update epochs. Discrete and continuous action spaces.
- **Interactive Plotly dashboards**: `reinforce.utils.plot_dashboard` renders a
  self-contained HTML dashboard (one panel per metric) from a `HistoryLogger`,
  a metrics dict, or a Logger CSV.
- **Prioritized Experience Replay + n-step for continuous control**: DDPG, TD3 and
  SAC now accept `prioritized=True` (with `per_alpha`/`per_beta_start`) and `n_step`,
  with importance-weighted critic losses and TD-error priority updates.
- **Rainbow DQN** (`Rainbow`): combines Double, Dueling, Prioritized Replay,
  n-step returns, distributional (C51) and Noisy Nets; new `NoisyLinear` /
  `RainbowNetwork` building blocks (C51's projection refactored for reuse).
- **Gymnasium vectorized training**: `make_gym_vec(id, num_envs, asynchronous)`
  vectorizes Gymnasium single envs with reinforce's own (correct-autoreset)
  vector envs for on-policy training.
- **Model-based RL**: `DynaQ` (tabular Dyna-Q — learned model + planning steps).
- **Episode GIF recording**: `render_rgb()` on CartPole/Pendulum/GridWorld/
  MountainCar and `reinforce.utils.record_gif`; animated agent GIFs in the README
  (`examples/record_gifs.py`).
- **Multi-agent RL** (`reinforce.multiagent`): `MultiAgentEnv` interface,
  `RockPaperScissors`, `CoordinationGame` and the multi-step cooperative
  `MultiAgentGridWorld`, and `MultiAgentPPO` supporting both shared-policy
  self-play and independent PPO (IPPO).
- **Recurrent PPO** (`RecurrentPPO`): LSTM actor-critic with proper hidden-state
  reset masking and truncated BPTT (minibatched over environments) for partially
  observable tasks; `BaseAgent.reset_states()` hook, called by `evaluate_policy`.
- **Environments**: `MountainCar`, `MountainCarContinuous`, `Acrobot`.
- **Dict observation spaces**: `core.spaces.Dict` + `flatten`/`flatdim` and the
  `FlattenDictObservation` wrapper (multi-modal observations).
- **Offline RL**: `IQL` (implicit Q-learning) and `CQL` (conservative Q-learning)
  in addition to TD3+BC.
- **Optuna** hyperparameter search (`reinforce.tuning.optuna_search`).
- **Weights & Biases** logging sink in `Logger`.
- Static type checking with **mypy** (config, CI step, badge); **Colab/Jupyter
  quickstart** notebook.
- **Developer experience**: `ProgressBarCallback` (live tqdm bar), `CheckpointCallback`
  (periodic saves), `EvalCallback` now saves the best model (`best_model_save_path`),
  `make_vec_env(...)` one-line vectorization, `anneal_lr` linear LR decay for PPO/A2C,
  and CLI `--version` / `--progress` / `--n-envs` / `--async`.
- **n-step returns** in `ReplayBuffer`/`PrioritizedReplayBuffer` (per-sample
  discount, correct across termination and truncation); `n_step` option on DQN
  and the continuous off-policy agents.
- **C51** and **QR-DQN** distributional value-based agents (`CategoricalQNetwork`,
  `QuantileQNetwork`).
- **Discrete SAC** (`SACDiscrete`): max-entropy off-policy for discrete actions.
- **Offline RL**: `TD3BC` (TD3 + behavior cloning) and `IQL` (implicit Q-learning),
  plus a `TransitionDataset` / `collect_dataset` data module.
- **Benchmark suite** (`examples/benchmark_scores.py`) reproducing scores for
  every algorithm; results table in `docs/benchmarks.md`.
- **CLI**: `reinforce train|eval|list` console script and `python -m reinforce`.
- **Registry & configs**: `make_agent` / `make_env`, per-(algo, env) tuned
  hyperparameters.
- **AsyncVectorEnv**: subprocess-based vectorized env (spawn) with the same API
  as `SyncVectorEnv`, for parallel data collection; `maybe_compile` torch.compile
  helper.
- **Pixel observations**: `CNNFeatureExtractor` + `ImageQNetwork`; DQN auto-uses a
  CNN for image `(C, H, W)` observations. Observation wrappers `FrameStack`,
  `FlattenObservation`, `OneHotObservation`.
- `py.typed` marker, pre-commit config, PyPI publish workflow (trusted publishing).
- **Documentation site** (MkDocs Material) with a GitHub Pages deploy workflow.

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
