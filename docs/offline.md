# Offline RL from logged data

In most real operations there is no simulator to explore in. What you have is a log of
past decisions and their outcomes, collected by whatever controller was running at the
time. Offline reinforcement learning trains a policy from that fixed log alone, with no
further interaction, which is what makes it deployable where online exploration is
expensive or unsafe.

[`examples/offline_rl_case_study.py`](https://github.com/DenisDrobyshev/decisionrl/blob/main/examples/offline_rl_case_study.py)
demonstrates this on the [`EnergyMicrogrid`](environments.md) battery-control task:

1. A mediocre controller runs the battery: the best greedy price-threshold rule with
   exploration noise added. Its transitions are recorded as a fixed dataset.
2. An offline agent (IQL or TD3+BC) is trained on that dataset only, using
   `learn_offline(dataset, steps)` with no environment access.
3. The learned policy is evaluated and compared against the behaviour policy that
   generated the log.

## Result

The behaviour policy that produced the log returns about 15.4. Trained on its logs alone,
offline IQL improves on it by about 10% (IQM over 5 seeds, reproduce with
`python examples/offline_rl_case_study.py --seeds 5 --algo iql`):

| Policy | Return (IQM [95% CI]) |
|---|---:|
| Behaviour policy (noisy greedy, logged) | 15.40 |
| Offline IQL (from logs only) | 16.88 [16.82, 16.93] |

The confidence interval sits entirely above the behaviour policy, so this is a robust
improvement, not seed noise. Offline RL recovers and improves on the good decisions buried
in a noisy log without ever touching the environment. The gain over the behaviour policy is
the value of learning from data you already have, rather than the value of online
exploration.

## Usage

```python
from decisionrl.data import collect_dataset
from decisionrl.algorithms import IQL
from decisionrl.envs import EnergyMicrogrid

dataset = collect_dataset(EnergyMicrogrid(), behaviour_policy, n_transitions=40_000)
agent = IQL(EnergyMicrogrid(), device="cuda")
agent.learn_offline(dataset, total_steps=40_000)
```

The dataset and the agent may live on different devices: `learn_offline` moves each
sampled batch to the agent's device, so a CPU-collected log can train a CUDA agent
directly. IQL, TD3+BC, and CQL all expose `learn_offline`.
