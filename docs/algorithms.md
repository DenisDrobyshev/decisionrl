# Algorithms

| Family | Algorithm | Class | Action space | Key features |
|---|---|---|---|---|
| Tabular | Q-Learning | `QLearning` | Discrete | off-policy TD |
| Tabular | SARSA | `SARSA` | Discrete | on-policy TD |
| Tabular | Expected SARSA | `ExpectedSARSA` | Discrete | lower-variance TD |
| Model-based | Dyna-Q | `DynaQ` | Discrete | learned model + planning |
| Model-based | MBPO | `MBPO` | Continuous | ensemble dynamics + short rollouts + SAC |
| Model-based | Dreamer (experimental) | `Dreamer` | Continuous | latent world model + actor-critic in imagination |
| Value-based | DQN | `DQN` | Discrete | Double · Dueling · PER · n-step · CNN |
| Value-based | C51 | `C51` | Discrete | distributional (categorical) DQN |
| Value-based | QR-DQN | `QRDQN` | Discrete | distributional (quantile regression) |
| Value-based | Rainbow | `Rainbow` | Discrete | Double + Dueling + PER + n-step + C51 + NoisyNets |
| Policy gradient | REINFORCE | `REINFORCE` | Discrete + Continuous | learned baseline |
| Actor-critic | A2C | `A2C` | Discrete + Continuous | GAE, vectorized |
| Actor-critic | PPO | `PPO` | Discrete + Continuous | clipped objective, GAE, KL early-stop |
| Actor-critic | GRPO | `GRPO` | Discrete + Continuous | critic-free, group-relative advantage (LLM-RLHF) |
| Actor-critic | IMPALA | `IMPALA` | Discrete + Continuous | V-trace off-policy correction, parallel actors |
| Actor-critic | Recurrent PPO | `RecurrentPPO` | Discrete | LSTM policy for partial observability (POMDPs) |
| Actor-critic | SAC (discrete) | `SACDiscrete` | Discrete | max-entropy, auto temperature |
| Continuous | DDPG | `DDPG` | Continuous | deterministic policy, action noise |
| Continuous | TD3 | `TD3` | Continuous | twin critics, delayed updates, smoothing |
| Continuous | SAC | `SAC` | Continuous | max-entropy, auto temperature |
| Offline | TD3+BC | `TD3BC` | Continuous | learns from a fixed dataset |
| Offline | IQL | `IQL` | Continuous | expectile value + advantage-weighted policy |
| Offline | CQL | `CQL` | Continuous | conservative Q-learning (SAC backbone) |
| Offline | Decision Transformer | `DecisionTransformer` | Discrete + Continuous | return-conditioned sequence modeling (causal GPT) |

See [Benchmarks](benchmarks.md) for reproduced scores across all algorithms.

## RLHF & intrinsic motivation

- **RLHF** (`reinforce.rlhf`): learn a reward from preferences (`RewardModel`,
  `synthetic_preferences`, `train_reward_model`) and optimize any agent against it
  via `RewardModelWrapper`. Pairs with **GRPO**, the policy-optimization method
  used to align language models.
- **Curiosity** (`reinforce.exploration`): `RND` and `ICM` intrinsic rewards, added
  to any environment with `CuriosityWrapper` for exploration on sparse-reward tasks.

## Correctness details

- **terminated vs truncated.** Off-policy buffers store the `terminated` flag
  only, so bootstrapping targets are correct on time-limit truncation. On-policy
  rollouts augment the reward with `gamma * V(final_obs)` at truncated steps.
- **n-step returns.** The replay buffer aggregates n-step transitions with a
  per-sample discount, exact across termination and truncation.
- **GAE** for advantage estimation, advantage normalization, orthogonal init,
  gradient clipping and learning-rate-agnostic entropy tuning are on by default
  where appropriate.

## Choosing an algorithm

- Discrete actions, sample-efficient → **DQN** (add `n_step`, `dueling`,
  `prioritized`) or **C51**.
- Discrete or continuous, robust default → **PPO**.
- Continuous control, sample-efficient → **SAC** or **TD3**.
- Learning from a fixed dataset → **TD3BC**.
