# Case study: inventory control on real demand

Most of the applied environments are driven by synthetic demand generators. This case
study replaces the generator with a real dataset and asks the same question: does a
learned policy beat the best classical rule, and if so, why.

## The data

The demand series is quarterly US real personal consumption expenditure (PCE) from 1959
to 2009, a genuine aggregate demand quantity shipped with statsmodels and saved to
[`examples/data/us_consumption.csv`](https://github.com/DenisDrobyshev/decisionrl/blob/main/examples/data/us_consumption.csv).
Real consumption grows roughly five-fold across the five decades and is punctuated by
recessions, so the series has a strong trend and repeated shocks rather than a fixed
mean.

[`DatasetDemandInventory`](https://github.com/DenisDrobyshev/decisionrl/blob/main/src/decisionrl/envs/dataset_demand_inventory.py)
rescales the series onto a single product's demand range and replays it: each episode
starts at a random offset and steps forward, drawing Poisson arrivals around the
empirical level at each point.

## Why the classical rule struggles

The base-stock ("order up to S") policy is optimal for stationary demand. Here demand is
not stationary: an early-era episode sits near the bottom of the range and a late-era
episode near the top. A fixed order-up-to level must commit to one number, so it is too
low in the high-demand era and too high in the low-demand era. A policy that observes
recent demand can infer which era it is in and adjust its order-up-to level, which is
exactly the signal `DatasetDemandInventory` exposes in the observation.

## Result

The comparison is against the best fixed base-stock, found by exhaustive 1-D search (the
exact optimum within that family), and the learned policy is reported as an interquartile
mean with a 95% bootstrap confidence interval over 5 seeds.

| Policy | Return (IQM [95% CI]) |
|---|---:|
| Fixed base-stock (exhaustive) | 705.0 |
| Learned policy (PPO) | 744.6 [671.0, 792.0] |
| Adaptive tracking base-stock | 841.6 |

Three tiers, and the ordering is the honest point of the study. The learned policy
improves on the best fixed base-stock by about 6%. That gain is genuine but modest: the
confidence interval reaches just below the baseline, because one of the five seeds does
not clear it, so this is a central-tendency improvement rather than a guaranteed win on
every run.

Above both sits a hand-written adaptive rule
([`tracking_base_stock`](https://github.com/DenisDrobyshev/decisionrl/blob/main/src/decisionrl/baselines.py)):
order up to the recent demand (the EWMA the environment exposes) plus a small safety
buffer. It reaches 841.6, clearly ahead of the learned policy. This is worth stating
plainly: when the structure of the problem is known, a simple heuristic that encodes it
can beat a policy learned from scratch. Reinforcement learning earns its place when that
structure is not known in advance, or is too complex to hand-code; here, where the
structure is exactly "track recent demand", the explicit rule wins. Both, in turn, beat
the fixed base-stock, because on real trending demand no single order-up-to level fits
every era.

## Reproduce

```bash
python examples/real_data_case_study.py --seeds 5
```

The script writes `real_data_case_study.json` with the base-stock level, the per-seed
returns, and the confidence interval.
