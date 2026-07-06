# Multi-agent RL

`reinforce.multiagent` provides a small multi-agent layer: an environment
interface, two example games, and a PPO-based learner that runs either as
**self-play** (one shared policy) or as **independent PPO** (a policy per agent).

## Environment interface

```python
class MultiAgentEnv:
    agents: list[str]
    observation_spaces: dict[str, Space]
    action_spaces: dict[str, Space]
    def reset(seed=None) -> (obs: dict, info: dict)
    def step(actions: dict) -> (obs, rewards, terminateds, truncateds, info)
```

Built-in games: `RockPaperScissors` (two-player zero-sum), `CoordinationGame`
(cooperative, single-shot — all agents rewarded when they pick the same action),
and `MultiAgentGridWorld` (cooperative **multi-step** navigation — each agent must
reach its own target under a dense distance reward).

## Self-play

```python
from reinforce.multiagent import MultiAgentPPO, RockPaperScissors

agent = MultiAgentPPO(RockPaperScissors(), shared_policy=True, seed=0)
agent.learn(40_000)
print(agent.policy_probs("player_0", [0.0]))   # learned mixed strategy
```

A single policy controls every player and learns from all of their experience at
once (agents become parallel columns of one rollout buffer).

## Independent PPO (IPPO)

```python
from reinforce.multiagent import MultiAgentPPO, CoordinationGame

agent = MultiAgentPPO(CoordinationGame(), shared_policy=False, seed=0)
agent.learn(20_000)   # each agent has its own policy/value/buffer
```

Independent learners reliably solve the cooperative coordination game (converging
on a common action). Note that naive gradient self-play on Rock-Paper-Scissors
cycles rather than converging to the uniform Nash equilibrium — a well-known
property of the game, not a bug.
