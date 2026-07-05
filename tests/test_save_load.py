import numpy as np

from reinforce.algorithms import DQN, PPO, SAC, QLearning
from reinforce.envs import CartPole, GridWorld, PointMass


def test_qlearning_save_load(tmp_path, quiet_logger):
    env = GridWorld(rows=3, cols=3)
    agent = QLearning(env, seed=0, logger=quiet_logger)
    agent.learn(2000)
    path = str(tmp_path / "q.pkl")
    agent.save(path)

    loaded = QLearning.load(path, env=GridWorld(rows=3, cols=3))
    for s in range(env.n_states):
        assert agent.predict(s) == loaded.predict(s)


def test_dqn_save_load(tmp_path, quiet_logger):
    env = GridWorld(rows=4, cols=4, one_hot=True)
    agent = DQN(env, learning_starts=50, batch_size=16, buffer_size=1000, seed=0, logger=quiet_logger)
    agent.learn(300)
    path = str(tmp_path / "dqn.pt")
    agent.save(path)

    loaded = DQN.load(path, env=GridWorld(rows=4, cols=4, one_hot=True))
    obs = np.eye(16, dtype=np.float32)
    for i in range(16):
        assert agent.predict(obs[i]) == loaded.predict(obs[i])


def test_ppo_save_load(tmp_path, quiet_logger):
    agent = PPO(CartPole(), n_steps=64, batch_size=32, n_epochs=1, seed=0, logger=quiet_logger)
    agent.learn(200)
    path = str(tmp_path / "ppo.pt")
    agent.save(path)

    loaded = PPO.load(path, env=CartPole())
    obs, _ = CartPole().reset(seed=1)
    assert agent.predict(obs, deterministic=True) == loaded.predict(obs, deterministic=True)


def test_sac_save_load(tmp_path, quiet_logger):
    env = PointMass()
    agent = SAC(env, learning_starts=50, batch_size=16, seed=0, logger=quiet_logger)
    agent.learn(200)
    path = str(tmp_path / "sac.pt")
    agent.save(path)

    loaded = SAC.load(path, env=PointMass())
    obs, _ = PointMass().reset(seed=1)
    a1 = np.asarray(agent.predict(obs, deterministic=True))
    a2 = np.asarray(loaded.predict(obs, deterministic=True))
    np.testing.assert_allclose(a1, a2, atol=1e-5)
