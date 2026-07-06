# Getting started

## Your first agent

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

Every agent exposes the same four methods regardless of family:

| Method | Purpose |
|---|---|
| `predict(obs, deterministic=True)` | map an observation to an action |
| `learn(total_steps, callback=None)` | train for a number of environment steps |
| `save(path)` / `load(path, env=...)` | round-trip the agent to disk |

## Reproducibility

```python
from reinforce.utils import set_seed
set_seed(42, deterministic=True)   # seeds Python, NumPy and PyTorch
```

Every agent takes a `seed=`, every environment accepts `reset(seed=...)`, and
every buffer/space has its own seedable RNG.

## Vectorized training

```python
from reinforce.algorithms import PPO
from reinforce.envs import CartPole
from reinforce.wrappers import SyncVectorEnv, AsyncVectorEnv

# in-process:
venv = SyncVectorEnv([lambda: CartPole() for _ in range(8)])
# or parallel processes (factories must be picklable):
venv = AsyncVectorEnv([CartPole for _ in range(8)])

agent = PPO(venv, n_steps=256, seed=0).learn(200_000)
```

## Offline RL

```python
from reinforce.algorithms import TD3BC
from reinforce.data import collect_dataset
from reinforce.envs import PointMass

dataset = collect_dataset(PointMass(), behaviour_policy, n_transitions=20_000)
agent = TD3BC(PointMass(), seed=0)
agent.learn_offline(dataset, total_steps=10_000)   # no env interaction
```

## Callbacks & training utilities

```python
from reinforce.training import ProgressBarCallback, EvalCallback, CheckpointCallback, CallbackList

agent.learn(100_000, callback=CallbackList([
    ProgressBarCallback(),                              # live tqdm progress bar
    EvalCallback(eval_env, eval_freq=5000, best_model_save_path="best.pt"),
    CheckpointCallback(save_freq=20_000, save_dir="checkpoints"),
]))
```

On-policy agents (PPO/A2C) also support linear learning-rate annealing via
`anneal_lr=True`. Create vectorized envs in one call with
`make_vec_env("CartPole", n_envs=8, asynchronous=True)`.

Turn logged metrics into an interactive HTML dashboard (Plotly):

```python
from reinforce.utils import HistoryLogger, plot_dashboard
log = HistoryLogger()
agent = PPO(CartPole(), logger=log); agent.learn(50_000)
plot_dashboard(log, "dashboard.html")   # one interactive panel per metric
```

## Learning from pixels

DQN automatically uses a CNN when the observation space is an image `(C, H, W)`:

```python
from reinforce.algorithms import DQN
agent = DQN(my_image_env, features_dim=64, seed=0).learn(50_000)
```
