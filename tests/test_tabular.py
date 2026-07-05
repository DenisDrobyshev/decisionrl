import pytest

from reinforce.algorithms import SARSA, ExpectedSARSA, QLearning
from reinforce.envs import GridWorld
from reinforce.training import evaluate_policy

ALGOS = [QLearning, SARSA, ExpectedSARSA]


def greedy_rollout(agent, env):
    obs, _ = env.reset(seed=0)
    steps, terminated, done = 0, False, False
    while not done and steps < 500:
        action = agent.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        steps += 1
        done = terminated or truncated
    return steps, terminated


@pytest.mark.parametrize("cls", ALGOS)
def test_tabular_constructs_and_predicts(cls, quiet_logger):
    agent = cls(GridWorld(rows=3, cols=3), seed=0, logger=quiet_logger)
    action = agent.predict(0)
    assert 0 <= action < 4


@pytest.mark.slow
@pytest.mark.parametrize("cls", ALGOS)
def test_tabular_learns_optimal_path(cls, quiet_logger):
    env = GridWorld(rows=4, cols=4, start=(0, 0), goal=(3, 3))
    agent = cls(env, learning_rate=0.2, gamma=0.99, seed=0, logger=quiet_logger)
    agent.learn(30_000)

    optimal = (env.rows - 1) + (env.cols - 1)  # Manhattan distance = 6
    steps, reached = greedy_rollout(agent, GridWorld(rows=4, cols=4, start=(0, 0), goal=(3, 3)))
    assert reached, f"{cls.__name__} greedy policy did not reach the goal"
    assert steps == optimal, f"{cls.__name__} path length {steps} != optimal {optimal}"

    mean_return, _ = evaluate_policy(agent, GridWorld(rows=4, cols=4), n_episodes=5)
    assert mean_return > 0.9
