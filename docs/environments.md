# Environments

All environments follow the Gymnasium API
(`reset(seed=...) -> (obs, info)`, `step(action) -> (obs, reward, terminated,
truncated, info)`) and need no external dependencies.

## Classic control

| Env | Class | Obs | Action |
|---|---|---|---|
| Grid navigation | `GridWorld` | Discrete or one-hot Box | Discrete(4) |
| Multi-armed bandit | `MultiArmedBandit` | Box(1) | Discrete(k) |
| CartPole | `CartPole` | Box(4) | Discrete(2) |
| Pendulum swing-up | `Pendulum` | Box(3) | Box(1) |
| Point-mass reach | `PointMass` | Box(n) | Box(n) |
| Mountain Car | `MountainCar` | Box(2) | Discrete(3) |
| Mountain Car (continuous) | `MountainCarContinuous` | Box(2) | Box(1) |
| Acrobot | `Acrobot` | Box(6) | Discrete(3) |

## Applied

| Env | Class | Problem |
|---|---|---|
| Inventory management | `InventoryManagement` | order under stochastic demand (operations) |
| Thermostat / HVAC | `Thermostat` | track a setpoint with minimal energy (control) |

See the [applied solutions](https://github.com/DenisDrobyshev/reinforce#applied-solutions)
in the README for trained results.

## Gymnasium interop (optional)

```python
from reinforce.envs import make_gym
env = make_gym("CartPole-v1")     # requires: pip install "reinforce[gym]"
```

Gymnasium environments already match this library's API, so agents consume them
directly; `make_gym` simply wraps one with reinforce's `Box`/`Discrete` spaces.

## Wrappers

`TimeLimit`, `NormalizeObservation`, `NormalizeReward`, `FrameStack`,
`FlattenObservation`, `OneHotObservation`, `SyncVectorEnv`, `AsyncVectorEnv`.
