# Robustness and stress testing

A policy that looks good on the environment it was tuned for can fail once the world
shifts. [`stress_test`](https://github.com/DrobyshevDev/decisionrl/blob/main/src/decisionrl/evaluation.py)
evaluates a policy across a set of perturbed environments and reports the mean return on
each, so a robust policy and a brittle one can be told apart.

```python
from decisionrl.evaluation import stress_test
from decisionrl.envs import NonstationaryInventory
from decisionrl import baselines as B

variants = {
    "nominal":      NonstationaryInventory,
    "demand_spike": lambda: NonstationaryInventory(demand_low=8.0, demand_high=24.0),
    "volatile":     lambda: NonstationaryInventory(switch_prob=0.20),
}
returns = stress_test(B.tracking_base_stock(3.0), variants, episodes=30)
```

Pass any `(env, obs) -> action` policy. To stress-test a trained agent, wrap it:
`lambda env, obs: agent.predict(obs)`.

## Example: fixed vs adaptive inventory policy

Mean return under three perturbations (the fixed base-stock is tuned on the nominal
setting):

| Variant | Fixed base-stock | Adaptive tracking |
|---|---:|---:|
| nominal | 268.9 | 322.4 |
| demand spike (higher demand) | 342.6 | 476.9 |
| volatile (fast regime switching) | 238.9 | 231.4 |

The adaptive policy is far more robust to a demand-level shift, which is exactly what it
is built for: it reads recent demand and follows it. Under very fast regime switching it
is marginally worse, because the smoothed demand estimate lags the rapid changes. That
trade-off is the point of a stress test: it surfaces where a policy holds up and where it
does not, instead of reporting a single number from the setting it was tuned on.
