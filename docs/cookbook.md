# Applied RL cookbook

A practical recipe for taking a real operational decision — "how much to reorder?",
"what price?", "admit or shed?" — and turning it into something a `decisionrl` agent
can solve *and* something you can trust. The emphasis is on the honest part:
**always measure against the classical baseline**, and know when *not* to use RL.

## The recipe in five steps

### 1. Frame the decision as an MDP

| Question | What it becomes |
|---|---|
| What do I observe when I decide? | **observation** (`observation_space`) |
| What do I choose? | **action** (`Discrete` for counts/choices, `Box` for continuous quantities) |
| What am I trading off? | **reward** (revenue − costs − penalties, per step) |
| When does an episode end? | **horizon** (truncation) or a terminal condition |

Keep the reward the *real* objective (profit, cost, service level), not a shaped
proxy — the whole point of applied RL is optimising the thing you actually care about.
Put any signal the optimal policy needs into the observation (e.g. a recent-demand
EWMA if demand drifts), otherwise the problem is needlessly partially observed.

### 2. Pick the baseline *first* (the honest part)

Before training anything, decide what "good" means. `decisionrl.baselines` and
`decisionrl.solvers` ship the classics:

```python
from decisionrl import baselines as B, solvers
from decisionrl.envs import InventoryManagement

# exact optimum where the problem is a small stationary MDP:
opt = solvers.inventory_optimal_value(InventoryManagement)          # value iteration
# strong heuristic baselines elsewhere:
_, best_bs = B.best_base_stock(InventoryManagement)                 # best order-up-to level
_, best_thr = B.best_value_threshold                                # (queues) best admission threshold
```

Compare against the *strongest* classical rule you can, not a straw man. Beating
"do nothing" proves nothing to a practitioner.

### 3. Choose the algorithm

| Situation | Reach for |
|---|---|
| Discrete action, needs stability | `DQN` (value-based) |
| Discrete or continuous, robust default | `PPO` |
| Continuous control, sample-efficient | `SAC`, `TD3` |
| Partial observability (history matters) | `RecurrentPPO` |
| Non-stationary / drifting | any of the above **+ a recent-signal feature in the observation** |

Value-based methods (`DQN`) are often steadier than policy-gradient on small discrete
operational problems; policy-gradient shines on continuous multi-dimensional actions.
(If a method collapses on some seeds, switch families and re-verify — don't ship a
fragile number.)

### 4. Verify honestly — multiple seeds, mean ± std

One seed is a rumour. Train across a few seeds and report the spread, against the
baseline from step 2. `examples/verify_applied_claims.py` is the template:

```python
from decisionrl.training import evaluate_policy
vals = []
for seed in range(3):
    agent = PPO(env_fn(), seed=seed).learn(50_000)
    vals.append(evaluate_policy(agent, env_fn(), n_episodes=20, seed=100)[0])
# report mean(vals) ± std(vals) vs the baseline
```

Record provenance with `decisionrl.tracking.run_manifest` (git SHA, versions, seed,
config, metrics) so the number is reproducible later.

### 5. Know when *not* to use RL

If the problem is **stationary and fully observed**, a formula or solver is usually
better — interpretable and provably optimal. `decisionrl` is honest about this: on
stationary inventory the learned policy only *matches* the DP optimum. RL earns its
place when those assumptions break:

- **non-stationary** dynamics (drifting demand, regime switches),
- **partial observability** (you don't see the full state),
- **coupled decisions** with no closed form (e.g. joint pricing + inventory),
- dynamics you **can't write as a clean LP/DP**.

That's the boundary the [proof table](https://github.com/DenisDrobyshev/decisionrl#why-decisionrl)
draws.

## Worked example: inventory in ~20 lines

```python
from decisionrl.algorithms import PPO
from decisionrl.envs import InventoryManagement
from decisionrl.training import evaluate_policy
from decisionrl import baselines as B, solvers

env_fn = InventoryManagement
agent = PPO(env_fn(), n_steps=1024, batch_size=64, n_epochs=10, seed=0).learn(40_000)

learned = evaluate_policy(agent, env_fn(), n_episodes=40)[0]
optimum = solvers.inventory_optimal_value(env_fn)     # exact DP optimum
print(f"PPO {learned:.1f}  vs DP optimum {optimum:.1f}")   # ~ matches: RL found the optimum
```

## Checklist

- [ ] Reward is the real objective, not a proxy.
- [ ] Observation contains what the optimal policy needs (add a drift/recent signal if non-stationary).
- [ ] A **strong** classical baseline is defined *before* training.
- [ ] Results are reported as mean ± std over ≥3 seeds.
- [ ] You can state *why RL* here (which classical assumption breaks) — or you used the solver instead.
