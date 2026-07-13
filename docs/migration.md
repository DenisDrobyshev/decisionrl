# Migrating from Stable-Baselines3 / CleanRL

`reinforce` is designed to feel familiar if you come from either library:
Stable-Baselines3's one-line agent API, and CleanRL's readable single-purpose
implementations. This guide maps the common patterns.

## From Stable-Baselines3

| Stable-Baselines3 | reinforce |
|---|---|
| `from stable_baselines3 import PPO` | `from reinforce.algorithms import PPO` |
| `PPO("MlpPolicy", "CartPole-v1")` | `PPO(make_env("CartPole"))` or `PPO(make_gym("CartPole-v1"))` |
| `model.learn(total_timesteps=50_000)` | `agent.learn(50_000)` |
| `model.predict(obs, deterministic=True)` | `agent.predict(obs, deterministic=True)` |
| `model.save("ppo") / PPO.load("ppo")` | `agent.save("ppo.pt") / PPO.load("ppo.pt", env=env)` |
| `evaluate_policy(model, env)` | `from reinforce.training import evaluate_policy` |
| `make_vec_env("CartPole-v1", n_envs=8)` | `make_vec_env("CartPole", n_envs=8, asynchronous=True)` |
| `VecNormalize` | `NormalizeObservation` / `NormalizeReward` wrappers |
| `HerReplayBuffer` | `reinforce.algorithms.HERDQN` (+ a goal env) |
| `tensorboard_log=...` | `Logger(tensorboard_dir=...)` (also CSV / W&B / Plotly) |

```python
# SB3
from stable_baselines3 import SAC
model = SAC("MlpPolicy", "Pendulum-v1", verbose=0)
model.learn(50_000)

# reinforce
from reinforce.algorithms import SAC
from reinforce.envs import Pendulum
agent = SAC(Pendulum(), seed=0).learn(50_000)
```

Key differences: agents take an **env instance** (not a policy string); `predict`
returns just the action (no hidden-state tuple) — use `reset_states()` for
recurrent agents; models save to an explicit file path and `load` takes `env=`.

## From CleanRL

CleanRL is single-file scripts; `reinforce` keeps the same readable update logic
but behind a reusable class, so you get CleanRL-level clarity **and** composition:

```python
# Instead of copying ppo_continuous.py and editing globals:
from reinforce.algorithms import PPO
from reinforce.envs import make_gym_vec

venv = make_gym_vec("CartPole-v1", num_envs=8, asynchronous=True)
agent = PPO(venv, n_steps=128, learning_rate=2.5e-4, seed=1).learn(500_000)
```

The correctness details CleanRL is careful about — GAE, advantage normalization,
terminated-vs-truncated bootstrapping, orthogonal init, gradient clipping — are all
on by default (see [Algorithms → correctness details](algorithms.md)).

## Gymnasium interop

Any Gymnasium env works directly via `make_gym` / `make_gym_vec`, and there are
convenience builders for common benchmarks: `make_atari` (DQN preprocessing),
`make_minigrid`, and `reinforce.multiagent.make_pettingzoo`.
