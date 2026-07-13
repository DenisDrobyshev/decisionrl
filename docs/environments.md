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

## Complex scenarios

Higher-dimensional, harder tasks spanning distinct domains — richer observations,
non-linear dynamics and real credit-assignment / exploration challenges.

| Env | Class | Obs | Action | Domain |
|---|---|---|---|---|
| Two-link reaching arm | `ReacherArm` | Box(10) | Box(2) | robotic manipulation |
| 2-D maze navigation (lidar) | `Navigation2D` | Box(14) | Box(2) | navigation / hard exploration |
| Lunar lander | `LunarLander` | Box(8) | Discrete(4) | rocket soft-landing control |
| Portfolio allocation | `PortfolioAllocation` | Box(4·n) | Box(n) | finance / sequential allocation |

- **ReacherArm** — 2-DoF torque-controlled arm; dense negative-distance reward with
  a control-effort penalty. Non-linear kinematics; solve with SAC / TD3.
- **Navigation2D** — point robot with momentum and lidar range sensors must reach a
  goal past walls with gaps; progress reward, collision penalty, terminal bonus.
  Pairs well with the `reinforce.exploration` curiosity bonuses.
- **LunarLander** — self-contained 2-D rigid-body lander; potential-based shaping
  (distance, speed, tilt, legs, fuel) plus a large land/crash terminal reward.
- **PortfolioAllocation** — allocate across correlated assets with AR(1) *momentum*
  returns and transaction costs; recent returns are predictive, so the optimal
  policy is not static and should beat equal-weight.

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

For vectorized Gymnasium training use `make_gym_vec`:

```python
from reinforce.envs import make_gym_vec
from reinforce.algorithms import PPO

venv = make_gym_vec("CartPole-v1", num_envs=8, asynchronous=True)
PPO(venv, n_steps=256, seed=0).learn(200_000)
```

It vectorizes Gymnasium *single* envs with reinforce's own vector envs, which use
correct immediate-autoreset and `final_observation` bootstrapping — stable across
Gymnasium autoreset-API changes.

For Atari, `make_atari` applies the standard DQN preprocessing (grayscale, resize
to 84×84, frame-skip, 4-frame stack) — ready for the built-in CNN:

```python
from reinforce.envs import make_atari
from reinforce.algorithms import DQN

agent = DQN(make_atari("ALE/Pong-v5"), seed=0)   # needs: pip install "gymnasium[atari]" ale-py
```

`make_minigrid("MiniGrid-Empty-5x5-v0")` (needs `pip install minigrid`) wraps
MiniGrid navigation envs, and `reinforce.multiagent.make_pettingzoo(...)` (needs
`pip install pettingzoo`) adapts PettingZoo parallel envs to `MultiAgentEnv`.

## Wrappers

`TimeLimit`, `NormalizeObservation`, `NormalizeReward`, `FrameStack`,
`FlattenObservation`, `OneHotObservation`, `SyncVectorEnv`, `AsyncVectorEnv`.
