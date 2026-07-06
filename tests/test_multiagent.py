import numpy as np
import pytest

from reinforce.multiagent import CoordinationGame, MultiAgentPPO, RockPaperScissors


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
