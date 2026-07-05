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


def test_checkpoint_resume_dqn(tmp_path, quiet_logger):
    def make():
        return GridWorld(rows=4, cols=4, one_hot=True)

    agent = DQN(make(), learning_starts=50, batch_size=16, buffer_size=1000, seed=0, logger=quiet_logger)
    agent.learn(300)
    steps = agent.num_timesteps
    path = str(tmp_path / "ckpt.pt")
    agent.save_checkpoint(path)

    loaded = DQN.load_checkpoint(path, env=make())
    assert loaded.num_timesteps == steps
    obs = np.eye(16, dtype=np.float32)
    for i in range(16):
        assert agent.predict(obs[i]) == loaded.predict(obs[i])
    loaded.learn(100)  # can continue training from the restored step count
    assert loaded.num_timesteps == steps + 100


def test_checkpoint_resume_ppo(tmp_path, quiet_logger):
    agent = PPO(CartPole(), n_steps=64, batch_size=32, n_epochs=1, seed=0, logger=quiet_logger)
    agent.learn(128)
    steps = agent.num_timesteps
    path = str(tmp_path / "ppo_ckpt.pt")
    agent.save_checkpoint(path)

    loaded = PPO.load_checkpoint(path, env=CartPole())
    assert loaded.num_timesteps == steps
    obs, _ = CartPole().reset(seed=3)
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
