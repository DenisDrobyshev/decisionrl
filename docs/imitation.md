# Imitation learning

`reinforce.imitation` learns policies from **demonstrations** rather than rewards.

| Method | Class | Idea |
|---|---|---|
| Behavioral Cloning | `BC` | supervised action prediction from a demo dataset |
| DAgger | `DAgger` | roll out the policy, relabel visited states with an expert, aggregate, retrain |
| GAIL | `GAIL` | a discriminator separates expert vs policy transitions; the policy (PPO) is trained to fool it |

```python
from reinforce.imitation import BC, DAgger, GAIL, collect_expert_dataset
from reinforce.envs import CartPole

demos = collect_expert_dataset(CartPole(), expert_policy, n_transitions=4000, seed=0)

# Behavioral cloning — pure supervised imitation
bc = BC(CartPole(), seed=0)
bc.train(demos, n_iters=1500)

# DAgger — needs a queryable expert to fix compounding error
dagger = DAgger(CartPole(), seed=0)
dagger.learn_dagger(CartPole(), expert_policy, iterations=4)

# GAIL — adversarial imitation, no environment reward at all
gail = GAIL(CartPole(), demos, seed=0)
gail.learn(iterations=10)
```

On CartPole with a heuristic expert, BC and DAgger reach the maximum return (500);
GAIL matches the expert from demonstrations alone, never seeing a reward.
Complements the offline-RL agents (`TD3BC`, `IQL`, `CQL`, `DecisionTransformer`)
and the preference-based methods (`reinforce.rlhf`).
