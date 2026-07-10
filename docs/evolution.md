# Evolutionary & swarm optimization

`reinforce.evolution` provides gradient-free (black-box) optimizers under one
**ask / tell** interface, plus a neuroevolution agent that trains RL policies
without gradients.

```python
from reinforce.evolution import CEM, minimize
from reinforce.evolution.functions import rastrigin

opt = CEM(dim=10, bounds=(-5.12, 5.12), seed=0)
x_best, f_best, history = minimize(rastrigin, opt, iters=200)
```

## Optimizers

| Family | Class | Idea |
|---|---|---|
| Evolution strategy | `CEM` | fit a Gaussian to the elite fraction |
| Evolution strategy | `CMAES` | covariance-matrix adaptation ES |
| Evolution strategy | `OpenAIES` | natural ES with mirrored sampling |
| Evolution strategy | `ARS` | augmented random search (V1-t) |
| Evolutionary | `DifferentialEvolution` | DE/rand/1/bin |
| Evolutionary | `GeneticAlgorithm` | tournament + blend + Gaussian mutation |
| Metaheuristic | `SimulatedAnnealing` | Metropolis acceptance + cooling |
| Swarm | `PSO` | particle swarm optimization |
| Swarm | `FireflyAlgorithm` | attraction to brighter fireflies |
| Swarm | `ArtificialBeeColony` | employed + scout bees |
| Swarm | `GreyWolfOptimizer` | alpha/beta/delta leader hierarchy |
| Swarm | `BatAlgorithm` | echolocation, loudness / pulse-rate |
| Combinatorial | `AntColonyTSP` | pheromone-guided TSP tours |

All continuous optimizers **minimize** the objective and share the ask/tell API,
so they are drop-in interchangeable and easy to benchmark.

![Gradient-free optimizers on Rastrigin](assets/evolution_benchmark.png)

## Neuroevolution

`NeuroevolutionAgent` optimizes a small tanh-MLP policy's weights directly to
maximize episode return using any optimizer above — no gradients, no replay
buffer. It implements the standard `predict / learn / save / load` API.

```python
from reinforce.evolution import NeuroevolutionAgent
from reinforce.envs import CartPole

agent = NeuroevolutionAgent(CartPole(), optimizer="cmaes", hidden_sizes=(16,), seed=0)
agent.learn(60_000)   # CEM / CMA-ES / PSO all reach return 500
```

![Neuroevolution on CartPole](assets/evolution_neuroevolution.png)

## Ant Colony Optimization (TSP)

```python
from reinforce.evolution import AntColonyTSP, random_cities, distance_matrix

cities = random_cities(20, seed=3)
tour, length, history = AntColonyTSP(distance_matrix(cities), seed=0).solve(iters=120)
```

![ACO for the TSP](assets/evolution_aco_tsp.png)
