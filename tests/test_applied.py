"""Learning tests for the applied environments (inventory, thermostat)."""

import numpy as np
import pytest

from reinforce.algorithms import PPO, SAC
from reinforce.envs import InventoryManagement, Thermostat
from reinforce.training import evaluate_policy


def _policy_return(env_fn, policy, episodes=20, seed=1):
    returns = []
    for ep in range(episodes):
        env = env_fn()
        obs, _ = env.reset(seed=seed + ep)
        done, total = False, 0.0
        while not done:
            obs, r, term, trunc, _ = env.step(policy(env, obs))
            total += r
            done = term or trunc
        returns.append(total)
    return float(np.mean(returns))


def test_applied_envs_construct_with_agents(quiet_logger):
    PPO(InventoryManagement(), n_steps=32, seed=0, logger=quiet_logger)
    SAC(Thermostat(), learning_starts=10, batch_size=8, seed=0, logger=quiet_logger)


@pytest.mark.slow
def test_ppo_solves_inventory(quiet_logger):
    agent = PPO(InventoryManagement(), n_steps=1024, batch_size=64, n_epochs=10,
                ent_coef=0.01, seed=0, logger=quiet_logger)
    agent.learn(30_000)
    learned = evaluate_policy(agent, InventoryManagement(), n_episodes=20, seed=1)[0]
    random_ret = _policy_return(InventoryManagement, lambda e, o: e.action_space.sample())
    assert learned > random_ret + 15, f"PPO inventory: learned={learned:.1f} vs random={random_ret:.1f}"


@pytest.mark.slow
def test_sac_controls_thermostat(quiet_logger):
    agent = SAC(Thermostat(), learning_starts=1000, batch_size=256, seed=0, logger=quiet_logger)
    agent.learn(12_000)
    learned = evaluate_policy(agent, Thermostat(), n_episodes=20, seed=1)[0]
    bang_bang = _policy_return(
        Thermostat, lambda e, o: np.array([1.0 if o[0] < 0 else -1.0], np.float32)
    )
    assert learned > bang_bang + 100, f"SAC thermostat: learned={learned:.1f} vs bang-bang={bang_bang:.1f}"
