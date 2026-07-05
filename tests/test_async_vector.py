"""Tests for the subprocess-based AsyncVectorEnv.

Factories must be picklable (spawn start method), so we pass the env class or a
functools.partial - never a lambda.
"""

from functools import partial

import numpy as np

from reinforce.envs import CartPole, PointMass
from reinforce.wrappers import AsyncVectorEnv


def test_async_vector_reset_and_step():
    venv = AsyncVectorEnv([CartPole, CartPole])
    try:
        assert venv.num_envs == 2
        assert venv.single_observation_space.shape == (4,)
        obs, _ = venv.reset(seed=0)
        assert obs.shape == (2, 4)
        actions = [venv.single_action_space.sample() for _ in range(2)]
        obs, rewards, terminateds, truncateds, infos = venv.step(actions)
        assert obs.shape == (2, 4)
        assert rewards.shape == (2,)
        assert terminateds.shape == (2,) and truncateds.shape == (2,)
    finally:
        venv.close()


def test_async_vector_autoreset_final_observation():
    venv = AsyncVectorEnv([partial(PointMass, max_steps=2), partial(PointMass, max_steps=2)])
    try:
        venv.reset(seed=0)
        saw_final = False
        for _ in range(6):
            _, _, terminateds, truncateds, infos = venv.step([np.zeros(2, np.float32)] * 2)
            if "final_observation" in infos:
                saw_final = True
                for i in range(2):
                    if terminateds[i] or truncateds[i]:
                        assert infos["final_observation"][i] is not None
        assert saw_final
    finally:
        venv.close()
