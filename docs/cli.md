# Command-line interface

Installing the package provides a `reinforce` console command (and
`python -m reinforce`).

```bash
reinforce list                                          # algorithms & environments
reinforce train ppo CartPole --steps 50000 --save ppo.pt
reinforce train dqn CartPole --set learning_rate=5e-4 --set buffer_size=100000
reinforce eval ppo --env CartPole --load ppo.pt --episodes 20
reinforce train ppo gym:LunarLander-v2 --steps 200000   # any Gymnasium env
```

## Tuned defaults

`train` applies tuned default hyperparameters per `(algorithm, environment)`
automatically. Override any of them with repeated `--set KEY=VALUE`, or disable
them with `--no-tuned`.

## Programmatic registry

The same string-based construction is available in code:

```python
from reinforce import make_env, make_agent, list_algorithms

print(list_algorithms())
agent = make_agent("ppo", make_env("CartPole"), seed=0)
agent.learn(50_000)
```
