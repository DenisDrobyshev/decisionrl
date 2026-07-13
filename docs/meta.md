# Meta-RL (RL²)

`decisionrl.meta` implements **RL²** (Duan et al., 2016; Wang et al., 2016) — meta-
reinforcement learning where a *recurrent* policy is trained across a whole
distribution of tasks so that its hidden state performs online adaptation. At test
time there are **no gradient steps**: the recurrent dynamics *are* the learning
algorithm the agent discovered.

```python
from decisionrl.algorithms import RecurrentPPO
from decisionrl.meta import make_meta_bandit
from decisionrl.wrappers import SyncVectorEnv

# a distribution of 5-armed Bernoulli bandits, 30 pulls per trial
venv = SyncVectorEnv([lambda i=i: make_meta_bandit(n_arms=5, horizon=30, seed=i)
                      for i in range(32)])
agent = RecurrentPPO(venv, n_steps=30, gae_lambda=0.3, seed=0).learn(500_000)

# on a *fresh* bandit the policy explores early then locks onto the best arm,
# purely from its recurrent state — no test-time training.
env = make_meta_bandit(n_arms=5, horizon=30, seed=999)
obs, _ = env.reset(); agent.reset_states()
for _ in range(30):
    obs, reward, _, done, info = env.step(agent.predict(obs, deterministic=False))
```

## How it works

The whole trick lives in the environment. [`RL2Env`](#rl2env) turns a task
distribution into a **single-trial** environment:

- **Task per trial.** Each `reset()` samples a fresh task via `task_fn(rng)`.
- **Augmented observation.** The policy sees the previous action (one-hot), the
  previous reward and the previous done flag concatenated with the task
  observation — the inputs it needs to infer the task online.
- **One long episode.** Inner-episode terminations are hidden from the agent and
  auto-reset the *same* task; the trial ends (as a truncation) only after `horizon`
  steps. Because the trial is a single episode, `RecurrentPPO` resets its recurrent
  state **only between trials** — so experience accumulates across a trial and the
  agent can adapt within it.

Train any recurrent agent (here `RecurrentPPO`) to maximize the total trial reward
and it learns to *explore then exploit* inside each trial.

## API

### `RL2Env(task_fn, horizon, seed=None)`

A `Wrapper` over a task distribution. `task_fn(rng: np.random.Generator) -> Env`
returns a freshly sampled task with a **discrete** action space. The observation
space is the task's flattened observation plus `n_actions + 2` extra dimensions.

### `make_meta_bandit(n_arms=5, horizon=50, seed=None)`

An `RL2Env` over Bernoulli bandits whose arm probabilities `p_i ~ U(0, 1)` are
resampled each trial — the classic RL² benchmark. Uses
[`decisionrl.envs.BernoulliBandit`](environments.md).

## Result

On held-out 5-arm bandits a meta-trained policy pulls the best arm far more often
than chance (random = 1/5 = 20%), and its hit-rate climbs over the course of a
trial as it narrows down the best arm — a bandit algorithm found by gradient
descent.
