<div align="center">

<img src="docs/assets/banner.svg" alt="reinforce — a correctness-first reinforcement learning foundation" width="100%">

# reinforce

**A dependency-light, correctness-first reinforcement learning foundation.**

Readable like [CleanRL](https://github.com/vwxyzjn/cleanrl), composable like
[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3), and
batteries-included so it runs the moment you `pip install` it.

[![CI](https://github.com/DenisDrobyshev/reinforce/actions/workflows/ci.yml/badge.svg)](https://github.com/DenisDrobyshev/reinforce/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-mkdocs--material-blue.svg)](https://denisdrobyshev.github.io/reinforce/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-orange.svg)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-blue.svg)](https://mypy-lang.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DenisDrobyshev/reinforce/blob/main/examples/quickstart.ipynb)

[![Algorithms](https://img.shields.io/badge/algorithms-25-8A2BE2.svg)](docs/algorithms.md)
[![Optimizers](https://img.shields.io/badge/gradient--free%20optimizers-12-9333ea.svg)](docs/evolution.md)
[![Tests](https://img.shields.io/badge/tests-266-brightgreen.svg)](tests)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Docs site](https://img.shields.io/badge/docs-site-1f6feb.svg)](https://denisdrobyshev.github.io/reinforce/)

<em>Tabular · value-based · policy-gradient · actor-critic · continuous control · offline · model-based · multi-agent · distributed — one clean, tested, typed library.</em>

</div>

---

## Overview

```mermaid
flowchart LR
    subgraph Interaction
        Env["Environment<br/>(GridWorld · CartPole · Pendulum<br/>Inventory · Thermostat · Gymnasium)"]
        Agent["Agent<br/>predict / learn / save / load"]
        Env -- "obs, reward, done" --> Agent
        Agent -- "action" --> Env
    end
    Agent --> Buf["Buffers<br/>Replay · PER · n-step · Rollout (GAE)"]
    Net["Networks<br/>MLP · CNN · Dueling · Categorical<br/>Gaussian · Noisy · Ensemble"] --> Agent
    Buf --> Learn["Learner update<br/>TD · PPO/GAE · V-trace · SAC · CQL · imagination"]
    Learn -- "gradients" --> Net
    Agent --> Log["Logger → CSV · TensorBoard · W&B · Plotly dashboard"]

    Vec["Vector envs (sync / async / distributed actors)"] -.-> Env
    Tune["Optuna search · CLI · tuned configs"] -.-> Agent
```

## Algorithm zoo

```mermaid
mindmap
  root(("reinforce"))
    Tabular
      Q-Learning
      SARSA
      Expected SARSA
    Model-based
      Dyna-Q
      MBPO
      Dreamer exp
    Value-based
      DQN
      C51
      QR-DQN
      Rainbow
    Policy actor-critic
      REINFORCE
      A2C
      PPO
      GRPO
      IMPALA
      Recurrent PPO
      SAC-Discrete
    Continuous
      DDPG
      TD3
      SAC
    Offline
      TD3-BC
      IQL
      CQL
      Decision Transformer
    Multi-agent
      MAPPO
      self-play IPPO
```

---

## See it learn

Every figure below is produced by a **single command** —
[`python examples/benchmark.py`](examples/benchmark.py) — which trains four agents
on four applied tasks *from scratch on CPU* in a few minutes and renders the plots.

![Agents learning applied tasks](docs/assets/learning_curves.png)

| Task | Algorithm | Result |
|---|---|---|
| CartPole (balance) | PPO | solved — return **500 / 500** |
| GridWorld 5×5 (navigate) | DQN (Double + Dueling) | near-optimal — return **≈ 0.93** |
| Pendulum (swing-up) | SAC | improves from ≈ −1300 to ≈ −420 |
| GridWorld 4×4 (navigate) | Q-Learning | optimal — return **≈ 0.95** |

The tabular agent recovers the optimal navigation policy (every arrow flows to the goal):

<p align="center"><img src="docs/assets/gridworld_policy.png" width="360" alt="Learned GridWorld policy"></p>

## Watch trained agents

<p align="center">
  <img src="docs/assets/cartpole_ppo.gif" width="240" alt="PPO balancing CartPole">
  <img src="docs/assets/pendulum_sac.gif" width="200" alt="SAC swinging up Pendulum">
  <img src="docs/assets/gridworld_qlearning.gif" width="200" alt="Q-Learning navigating GridWorld">
</p>

<p align="center"><em>PPO balances CartPole · SAC swings up Pendulum · Q-Learning navigates GridWorld — all rendered by <code>python examples/record_gifs.py</code>.</em></p>

---

## Applied solutions

Beyond classic control, `reinforce` ships two **applied** environments that mirror
real decision problems. Reproduce everything below with
[`python examples/applied_demo.py`](examples/applied_demo.py).

### 📦 Inventory management (operations research)

An agent sets weekly re-order quantities under stochastic (Poisson) demand,
trading off holding cost, ordering cost and stockouts. **PPO recovers the optimal
base-stock policy from scratch** — it matches a hand-tuned analytic heuristic
(**≈ 197 vs 199** profit) and crushes a random policy (**≈ 160**), with zero
domain knowledge.

![Inventory management with PPO](docs/assets/applied_inventory.png)

### 🌡️ Thermostat / HVAC control (energy)

An agent modulates a heating/cooling unit to hold a room at its setpoint while the
outdoor temperature swings on a daily cycle. **SAC tracks the setpoint smoothly
using about ⅓ of the energy of a bang-bang thermostat** (return **≈ −40 vs −303**;
energy **≈ 59 vs 200**).

![Thermostat / HVAC control with SAC](docs/assets/applied_thermostat.png)

---

## Complex scenarios

Beyond the toy tasks, `reinforce` ships four **harder, higher-dimensional
environments** from distinct domains — richer observations, non-linear dynamics
and real exploration / credit-assignment challenges. Train them all with
[`python examples/complex_scenarios.py`](examples/complex_scenarios.py) (uses the
GPU automatically). Measured results (random start → after training):

| Scenario | Domain | Obs · Action | Algorithm | Return: before → after |
|---|---|---|---|---|
| 🦾 **ReacherArm** | robotic manipulation | Box(10) · Box(2) | SAC | −8.1 → **−5.6** |
| 🧭 **Navigation2D** (lidar maze) | navigation / hard exploration | Box(14) · Box(2) | SAC | −9.5 → **+5.0** (reaches goal) |
| 🚀 **LunarLander** | rocket soft-landing | Box(8) · Discrete(4) | PPO | −435 → **+79** (lands) |
| 📈 **PortfolioAllocation** | finance / allocation | Box(4n) · Box(n) | SAC | beats equal-weight (**+0.10 vs −0.02**) |

Each is self-contained (NumPy only, Gymnasium API) — see the
[environments docs](docs/environments.md). `Navigation2D` pairs naturally with the
[curiosity bonuses](#curiosity--return-conditioned-control) for sparse-reward
exploration.

---

## Why another RL library?

Most RL code forces a trade-off: either it is a single readable file you cannot
reuse, or it is a powerful framework you cannot read. `reinforce` aims for the
middle: **every algorithm is short and legible, but built from shared,
swappable components** (buffers, networks, policies, schedules, wrappers).

Three principles guide it:

1. **Correctness-first.** The subtle things that quietly break RL agents are
   handled properly — most notably the Gymnasium `terminated` vs `truncated`
   distinction, which is bootstrapped correctly everywhere (time-limit
   truncation bootstraps from the final observation; true termination does not).
   It also ships GAE, target-policy smoothing, automatic entropy tuning,
   orthogonal init, advantage normalization and observation/reward normalization.
2. **Dependency-light & batteries-included.** The core needs only **NumPy + PyTorch**.
   Built-in environments (GridWorld, bandit, CartPole, Pendulum, PointMass) mean
   you can train an agent end-to-end with *zero* extra installs. Gymnasium is an
   optional extra, not a requirement.
3. **One API for everything.** Tabular or deep, discrete or continuous,
   on-policy or off-policy — every agent exposes the same four methods:
   `predict` · `learn` · `save` · `load`.

## Installation

```bash
# from PyPI — distributed as "reinforce-rl", imported as "reinforce"
pip install reinforce-rl

# with Gymnasium environments
pip install "reinforce-rl[gym]"

# latest from source
pip install git+https://github.com/DenisDrobyshev/reinforce.git

# local dev install
git clone https://github.com/DenisDrobyshev/reinforce.git
cd reinforce
pip install -e ".[dev]"
```

> The import package is always `reinforce` (`import reinforce`); the PyPI
> distribution is named `reinforce-rl` because `reinforce` was already taken.

## Quick start

```python
from reinforce.algorithms import PPO
from reinforce.envs import CartPole
from reinforce.training import evaluate_policy
from reinforce.utils import set_seed

set_seed(0)
agent = PPO(CartPole(), n_steps=1024, seed=0)
agent.learn(total_steps=50_000)

mean, std = evaluate_policy(agent, CartPole(), n_episodes=20)
print(f"return = {mean:.1f} +/- {std:.1f}")

agent.save("ppo_cartpole.pt")
agent = PPO.load("ppo_cartpole.pt", env=CartPole())
```

Tabular control is just as simple:

```python
from reinforce.algorithms import QLearning
from reinforce.envs import GridWorld

agent = QLearning(GridWorld(rows=5, cols=5), seed=0)
agent.learn(total_steps=20_000)
```

Continuous control with SAC:

```python
from reinforce.algorithms import SAC
from reinforce.envs import Pendulum

agent = SAC(Pendulum(), seed=0)
agent.learn(total_steps=20_000)
```

Use a Gymnasium environment (optional extra):

```python
from reinforce.algorithms import PPO
from reinforce.envs import make_gym

agent = PPO(make_gym("CartPole-v1"), seed=0)
agent.learn(total_steps=100_000)
```

Scale on-policy training with vectorized environments:

```python
from reinforce.algorithms import PPO
from reinforce.envs import CartPole
from reinforce.wrappers import SyncVectorEnv

venv = SyncVectorEnv([lambda: CartPole() for _ in range(8)])
agent = PPO(venv, n_steps=256, seed=0)   # 8 x 256 = 2048 steps per update
agent.learn(total_steps=200_000)
```

## Command-line interface

Train and evaluate without writing a script — tuned default hyperparameters are
applied automatically per (algorithm, environment) and can be overridden:

```bash
reinforce list                                          # show algorithms & envs
reinforce train ppo CartPole --steps 50000 --save ppo.pt --progress
reinforce train dqn CartPole --set learning_rate=5e-4 --set buffer_size=100000
reinforce train ppo CartPole --n-envs 8 --async         # parallel data collection
reinforce eval ppo --env CartPole --load ppo.pt --episodes 20
reinforce train ppo gym:LunarLander-v2 --steps 200000   # any Gymnasium env
```

Programmatic equivalents via the registry:

```python
from reinforce import make_env, make_agent, make_vec_env
agent = make_agent("ppo", make_env("CartPole"), seed=0).learn(50_000)
venv = make_vec_env("CartPole", n_envs=8, asynchronous=True)   # one-line vectorization
```

## Training utilities & callbacks

```python
from reinforce.algorithms import PPO
from reinforce.envs import CartPole
from reinforce.training import ProgressBarCallback, EvalCallback, CheckpointCallback, CallbackList

agent = PPO(CartPole(), anneal_lr=True, seed=0)      # linear LR decay (best practice)
agent.learn(100_000, callback=CallbackList([
    ProgressBarCallback(),                            # live tqdm bar (steps/s, ETA, return)
    EvalCallback(CartPole(), eval_freq=5000, best_model_save_path="best.pt"),
    CheckpointCallback(save_freq=20_000, save_dir="checkpoints"),
]))
```

## Algorithms

| Family | Algorithm | Class | Action space | Key features |
|---|---|---|---|---|
| Tabular | Q-Learning | `QLearning` | Discrete | off-policy TD |
| Tabular | SARSA | `SARSA` | Discrete | on-policy TD |
| Tabular | Expected SARSA | `ExpectedSARSA` | Discrete | lower-variance TD |
| Model-based | Dyna-Q | `DynaQ` | Discrete | learned model + planning (Sutton 1990) |
| Model-based | MBPO | `MBPO` | Continuous | ensemble dynamics + short rollouts + SAC |
| Model-based | Dreamer* | `Dreamer` | Continuous | latent world model + imagination (*experimental) |
| Value-based | DQN | `DQN` | Discrete | Double · Dueling · PER · **n-step** · CNN |
| Value-based | C51 | `C51` | Discrete | distributional (categorical) DQN |
| Value-based | QR-DQN | `QRDQN` | Discrete | distributional (quantile regression) |
| Value-based | **Rainbow** | `Rainbow` | Discrete | Double+Dueling+PER+n-step+C51+NoisyNets |
| Policy gradient | REINFORCE | `REINFORCE` | Discrete + Continuous | learned baseline |
| Actor-critic | A2C | `A2C` | Discrete + Continuous | GAE, vectorized |
| Actor-critic | PPO | `PPO` | Discrete + Continuous | clipped objective, GAE, KL early-stop |
| Actor-critic | **GRPO** | `GRPO` | Discrete + Continuous | critic-free, group-relative advantage (LLM-RLHF) |
| Actor-critic | IMPALA | `IMPALA` | Discrete + Continuous | V-trace, parallel actors |
| Actor-critic | Recurrent PPO | `RecurrentPPO` | Discrete | LSTM policy for partial observability |
| Actor-critic | SAC (discrete) | `SACDiscrete` | Discrete | max-entropy, auto temperature |
| Continuous | DDPG | `DDPG` | Continuous | deterministic policy, noise · PER · n-step |
| Continuous | TD3 | `TD3` | Continuous | twin critics, delayed updates · PER · n-step |
| Continuous | SAC | `SAC` | Continuous | max-entropy, auto temperature · PER · n-step |
| Offline | TD3+BC | `TD3BC` | Continuous | learns from a fixed dataset (no env) |
| Offline | IQL | `IQL` | Continuous | expectile value + advantage-weighted policy |
| Offline | CQL | `CQL` | Continuous | conservative Q-learning (SAC backbone) |
| Offline | **Decision Transformer** | `DecisionTransformer` | Discrete + Continuous | return-conditioned sequence modeling (GPT) |

See [reproduced benchmark scores](docs/benchmarks.md) for all algorithms.

## RLHF & LLM-style alignment

The same recipe that aligns language models, on control tasks: learn a reward
from **preferences**, then optimize a policy against it with **GRPO** — the
critic-free method behind modern LLM RLHF.

```python
from reinforce.envs import PointMass
from reinforce.rlhf import collect_segments, synthetic_preferences, RewardModel, train_reward_model
from reinforce.algorithms import SAC

env = PointMass()
segments = collect_segments(env, lambda o: env.action_space.sample(), n_segments=120, seg_len=25, seed=0)
prefs = synthetic_preferences(segments, n_pairs=800, seed=1)          # a preference teacher

reward_model = RewardModel(obs_dim=2, action_space=env.action_space, use_action=False)
train_reward_model(reward_model, prefs, n_iters=500)                  # Bradley-Terry likelihood
# -> learned reward correlates ≈0.98 with the true reward; optimize any agent on it:
from reinforce.rlhf import RewardModelWrapper
agent = SAC(RewardModelWrapper(env, reward_model), seed=0).learn(20_000)
```

`reinforce.rlhf` provides `RewardModel`, `PreferenceDataset`, `collect_segments`,
`synthetic_preferences`, `train_reward_model` and `RewardModelWrapper`.

## Curiosity & return-conditioned control

```python
# Intrinsic motivation: any agent gets exploration on sparse-reward tasks for free.
from reinforce.exploration import RND, CuriosityWrapper
from reinforce.algorithms import DQN
env = CuriosityWrapper(CartPole(), RND(obs_dim=4))      # or ICM(...)
DQN(env, seed=0).learn(50_000)

# Decision Transformer: offline RL as return-conditioned sequence modeling.
from reinforce import collect_trajectories
from reinforce.algorithms import DecisionTransformer
data = collect_trajectories(CartPole(), policy, n_trajectories=150, seed=0)
dt = DecisionTransformer(CartPole(), seed=0).learn_offline(data, n_iters=2500)
dt.evaluate(CartPole(), target_return=500)             # condition on the return you want
```

## Evolutionary & swarm optimization

A full family of **gradient-free** optimizers under one ask/tell interface —
evolution strategies **CEM · CMA-ES · Differential Evolution · Genetic Algorithm ·
OpenAI-ES · ARS · Simulated Annealing** and swarm intelligence **PSO · Firefly ·
Artificial Bee Colony · Grey Wolf · Bat · Ant Colony (TSP)** — plus a
`NeuroevolutionAgent` that trains RL policies with any of them (no gradients).

```python
from reinforce.evolution import CEM, minimize
from reinforce.evolution.functions import rastrigin
x, f, history = minimize(rastrigin, CEM(dim=10, bounds=(-5.12, 5.12), seed=0), iters=200)

# Neuroevolution: solve CartPole with a gradient-free optimizer
from reinforce.evolution import NeuroevolutionAgent
from reinforce.envs import CartPole
agent = NeuroevolutionAgent(CartPole(), optimizer="cmaes", seed=0).learn(60_000)
```

All twelve optimizers converge on the multimodal Rastrigin function (Grey Wolf
reaches ~1e-9); every neuroevolution optimizer **solves CartPole (return 500) with
no gradients**; and Ant Colony Optimization finds short TSP tours. Reproduce with
[`python examples/evolution_demo.py`](examples/evolution_demo.py).

![Gradient-free optimizers on Rastrigin](docs/assets/evolution_benchmark.png)

![Neuroevolution on CartPole](docs/assets/evolution_neuroevolution.png)

![Ant Colony Optimization for the TSP](docs/assets/evolution_aco_tsp.png)

## Serving trained policies

Export any trained agent to **ONNX** (or TorchScript) and serve it over HTTP —
the serving image needs only `onnxruntime` + FastAPI, no PyTorch.

```python
from reinforce.algorithms import PPO
from reinforce.envs import CartPole
from reinforce.serving import export_onnx, OnnxPolicy

agent = PPO(CartPole(), seed=0).learn(50_000)
export_onnx(agent, "policy.onnx")            # + policy.onnx.json metadata
action = OnnxPolicy("policy.onnx").predict(obs)   # inference without torch
```

```bash
# FastAPI service (POST /predict, GET /health, GET /info) — see deploy/Dockerfile
REINFORCE_MODEL=policy.onnx uvicorn reinforce.serving.server:app
```

## Multi-agent

```python
from reinforce.multiagent import MultiAgentPPO, RockPaperScissors, CoordinationGame

# self-play (one shared policy controls every agent)
selfplay = MultiAgentPPO(RockPaperScissors(), shared_policy=True, seed=0).learn(40_000)

# independent PPO (a separate policy per agent) on a cooperative game
ippo = MultiAgentPPO(CoordinationGame(), shared_policy=False, seed=0).learn(20_000)
```

`reinforce.multiagent` adds a `MultiAgentEnv` interface, example games, and
`MultiAgentPPO` (self-play or IPPO). See the [multi-agent docs](docs/multiagent.md).

## Distributed training

```python
from reinforce import DistributedActorLearner
from reinforce.envs import CartPole

# real actor processes stream trajectories to a central V-trace learner
learner = DistributedActorLearner(CartPole, num_actors=8, seed=0).learn(200_000)
```

`DistributedActorLearner` is a true multi-process, IMPALA-style architecture: each
actor runs its own environment and local policy inference in a separate process,
and the learner performs V-trace updates and broadcasts fresh weights every
iteration.

## Components you can reuse

```
reinforce
├── core         # Env, Wrapper, Space (Box/Discrete), BaseAgent, Transition
├── envs         # GridWorld, MultiArmedBandit, CartPole, Pendulum, PointMass,
│                 #   InventoryManagement, Thermostat (applied), make_gym
├── buffers      # ReplayBuffer, PrioritizedReplayBuffer (sum-tree), RolloutBuffer (GAE)
├── networks     # build_mlp, QNetwork, Dueling/CategoricalQNetwork (C51),
│                 #   CNNFeatureExtractor + ImageQNetwork (pixels),
│                 #   Categorical/Gaussian/Squashed policies
├── exploration  # Linear/Exponential schedules, Gaussian & Ornstein-Uhlenbeck noise
├── wrappers     # TimeLimit, NormalizeObservation, NormalizeReward,
│                 #   SyncVectorEnv, AsyncVectorEnv (multiprocessing),
│                 #   FrameStack, FlattenObservation, OneHotObservation
├── utils        # set_seed, Logger (stdout/CSV/TensorBoard), RunningMeanStd, torch helpers
├── training     # evaluate_policy, Callback, EvalCallback, StopOnRewardThreshold
└── algorithms   # the ten agents above
```

Everything is duck-typed against the Gymnasium API, so `reinforce` components and
Gymnasium environments interoperate freely in either direction.

## Reproducibility

```python
from reinforce.utils import set_seed
set_seed(42, deterministic=True)   # seeds Python, NumPy, PyTorch (+ deterministic kernels)
```

Every agent accepts a `seed=` argument, every environment accepts `reset(seed=...)`,
and every buffer/space has its own seedable RNG.

## Development & testing

```bash
pip install -e ".[dev]"
pytest              # full suite (unit + integration "does it actually learn?" tests)
ruff check .        # lint
```

The test suite covers component correctness (spaces, buffers, sum-tree,
schedules, GAE, normalization, save/load round-trips) **and** learning behaviour:
tabular methods reach the optimal policy on GridWorld, DQN/PPO learn CartPole,
and SAC/TD3/DDPG solve the PointMass reaching task.

## Design notes

- **terminated vs truncated.** Off-policy buffers store the `terminated` flag
  only, so bootstrapping targets are correct on time-limit truncation. On-policy
  rollouts augment the reward with `gamma * V(final_obs)` at truncated steps and
  mark the episode boundary — the Stable-Baselines3 approach.
- **No hidden global state.** No registries, no config magic; you construct
  objects and call methods.
- **Small surface, deep correctness.** The goal is a foundation you can read in
  an afternoon and trust in a paper.

## License

[MIT](LICENSE) © 2026 Denis Drobyshev

## Acknowledgements

Inspired by the clarity of CleanRL, the API design of Stable-Baselines3, the
modularity of Tianshou, and the Farama Foundation's Gymnasium standard.
