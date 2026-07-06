import numpy as np
import pytest

from reinforce.multiagent import (
    CoordinationGame,
    MultiAgentGridWorld,
    MultiAgentPPO,
    RockPaperScissors,
)


def test_rps_payoff_and_api():
    env = RockPaperScissors()
    obs, _ = env.reset(seed=0)
    assert set(obs) == {"player_0", "player_1"}
    assert obs["player_0"].shape == (1,)
    # rock (0) beats scissors (2)
    _, rewards, term, trunc, _ = env.step({"player_0": 0, "player_1": 2})
    assert rewards["player_0"] == 1.0 and rewards["player_1"] == -1.0
    assert all(term.values())
    # tie
    _, rewards, _, _, _ = env.step({"player_0": 1, "player_1": 1})
    assert rewards["player_0"] == 0.0


def test_coordination_reward():
    env = CoordinationGame(n_agents=2, n_actions=3)
    env.reset(seed=0)
    _, r_same, _, _, _ = env.step({"agent_0": 2, "agent_1": 2})
    _, r_diff, _, _, _ = env.step({"agent_0": 0, "agent_1": 1})
    assert r_same["agent_0"] == 1.0 and r_diff["agent_0"] == 0.0


@pytest.mark.parametrize("shared", [True, False])
def test_mappo_constructs_and_predicts(shared, quiet_logger):
    agent = MultiAgentPPO(RockPaperScissors(), shared_policy=shared, n_steps=16, n_epochs=1,
                          seed=0, logger=quiet_logger)
    probs = agent.policy_probs("player_0", np.zeros(1, np.float32))
    assert probs.shape == (3,) and abs(probs.sum() - 1.0) < 1e-5
    assert agent.predict(np.zeros(1, np.float32), agent="player_0") in (0, 1, 2)


def test_mappo_selfplay_rps_smoke(quiet_logger):
    agent = MultiAgentPPO(RockPaperScissors(), shared_policy=True, n_steps=64, n_epochs=2,
                          seed=0, logger=quiet_logger)
    agent.learn(256)
    assert agent.num_timesteps >= 256


def _greedy_joint_reward(agent, make_env, episodes=50):
    rewards = []
    for _ in range(episodes):
        env = make_env()
        obs, _ = env.reset()
        actions = {a: agent.predict(obs[a], agent=a, deterministic=True) for a in env.agents}
        _, rew, _, _, _ = env.step(actions)
        rewards.append(rew[env.agents[0]])
    return float(np.mean(rewards))


@pytest.mark.slow
@pytest.mark.parametrize("shared", [True, False])
def test_mappo_learns_coordination(shared, quiet_logger):
    def make():
        return CoordinationGame(n_agents=2, n_actions=3)

    agent = MultiAgentPPO(make(), shared_policy=shared, n_steps=128, n_epochs=4, batch_size=64,
                          ent_coef=0.01, seed=0, logger=quiet_logger)
    agent.learn(20_000)
    joint = _greedy_joint_reward(agent, make)
    assert joint > 0.6, f"MultiAgentPPO (shared={shared}) failed to coordinate (joint={joint:.2f})"


def test_ma_gridworld_api():
    env = MultiAgentGridWorld(rows=4, cols=4, n_agents=2, horizon=8)
    obs, _ = env.reset(seed=0)
    assert set(obs) == set(env.agents)
    assert obs["agent_0"].shape == (4,)
    done = False
    for _ in range(8):  # all "stay" -> truncates at the horizon
        _, rewards, term, trunc, _ = env.step(dict.fromkeys(env.agents, 4))
        done = all(term.values()) or all(trunc.values())
    assert done


def _ma_episode_return(agent, make_env, episodes=20, seed=1):
    outs = []
    for ep in range(episodes):
        env = make_env()
        obs, _ = env.reset(seed=seed + ep)
        totals = dict.fromkeys(env.agents, 0.0)
        done = False
        while not done:
            actions = {a: agent.predict(obs[a], agent=a, deterministic=True) for a in env.agents}
            obs, rew, term, trunc, _ = env.step(actions)
            for a in env.agents:
                totals[a] += rew[a]
            done = all(term.values()) or all(trunc.values())
        outs.append(np.mean(list(totals.values())))
    return float(np.mean(outs))


@pytest.mark.slow
@pytest.mark.parametrize("shared", [True, False])
def test_mappo_learns_gridworld_navigation(shared, quiet_logger):
    def make():
        return MultiAgentGridWorld(rows=5, cols=5, n_agents=2, horizon=25)

    agent = MultiAgentPPO(make(), shared_policy=shared, n_steps=256, n_epochs=4, batch_size=64,
                          ent_coef=0.01, seed=0, logger=quiet_logger)
    agent.learn(30_000)
    ret = _ma_episode_return(agent, make)
    assert ret > -5.0, f"MultiAgentPPO (shared={shared}) failed to navigate (return={ret:.2f})"
