"""Image observations: CNN extractor, DQN-on-pixels, and observation wrappers."""

from typing import Optional

import numpy as np
import pytest
import torch

from decisionrl.algorithms import DQN
from decisionrl.core.env import Env
from decisionrl.core.spaces import Box, Discrete
from decisionrl.envs import GridWorld
from decisionrl.networks import CNNFeatureExtractor, ImageQNetwork, is_image_space
from decisionrl.training import evaluate_policy
from decisionrl.wrappers import FlattenObservation, FrameStack, OneHotObservation

_DELTAS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


class ImageReacher(Env):
    """Tiny image env: navigate a dot (channel 0) to a fixed target (channel 1)."""

    def __init__(self, size: int = 6, max_steps: int = 40) -> None:
        self.size = size
        self.max_steps = max_steps
        self.observation_space = Box(0.0, 1.0, shape=(2, size, size), dtype=np.float32)
        self.action_space = Discrete(4)
        self.target = (size // 2, size // 2)
        self._rng = np.random.default_rng()
        self._pos = (0, 0)
        self._steps = 0

    def _obs(self) -> np.ndarray:
        img = np.zeros((2, self.size, self.size), dtype=np.float32)
        img[0, self._pos[0], self._pos[1]] = 1.0
        img[1, self.target[0], self.target[1]] = 1.0
        return img

    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._pos = (int(self._rng.integers(self.size)), int(self._rng.integers(self.size)))
        self._steps = 0
        return self._obs(), {}

    def step(self, action):
        dr, dc = _DELTAS[int(action)]
        self._pos = (
            int(np.clip(self._pos[0] + dr, 0, self.size - 1)),
            int(np.clip(self._pos[1] + dc, 0, self.size - 1)),
        )
        self._steps += 1
        reached = self._pos == self.target
        reward = 1.0 if reached else -0.05
        truncated = self._steps >= self.max_steps and not reached
        return self._obs(), reward, reached, truncated, {}


def test_is_image_space():
    assert is_image_space(Box(0, 1, shape=(3, 8, 8)))
    assert not is_image_space(Box(0, 1, shape=(4,)))
    assert not is_image_space(Discrete(4))


def test_cnn_feature_extractor_shape():
    cnn = CNNFeatureExtractor((2, 8, 8), features_dim=32)
    out = cnn(torch.zeros(5, 2, 8, 8))
    assert out.shape == (5, 32)


def test_image_qnetwork_shape():
    net = ImageQNetwork((2, 6, 6), n_actions=4, features_dim=32)
    assert net(torch.zeros(3, 2, 6, 6)).shape == (3, 4)


def test_frame_stack_image_and_vector():
    fs = FrameStack(ImageReacher(size=6), k=3)
    assert fs.observation_space.shape == (6, 6, 6)  # 3 * 2 channels
    obs, _ = fs.reset(seed=0)
    assert obs.shape == (6, 6, 6)

    fs2 = FrameStack(GridWorld(one_hot=True), k=2)
    obs, _ = fs2.reset(seed=0)
    assert obs.shape == (2 * 16,)


def test_flatten_observation():
    env = FlattenObservation(ImageReacher(size=6))
    assert env.observation_space.shape == (2 * 6 * 6,)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (72,)


def test_one_hot_observation():
    env = OneHotObservation(GridWorld(rows=3, cols=3))  # Discrete(9) obs
    assert env.observation_space.shape == (9,)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (9,) and obs.sum() == 1.0


def test_dqn_image_constructs(quiet_logger):
    agent = DQN(ImageReacher(), features_dim=32, learning_starts=10, batch_size=8,
                buffer_size=500, seed=0, logger=quiet_logger)
    assert agent.is_image
    obs, _ = ImageReacher().reset(seed=0)
    assert 0 <= agent.predict(obs) < 4


@pytest.mark.slow
def test_dqn_learns_from_pixels(quiet_logger):
    def make():
        return ImageReacher(size=6, max_steps=40)

    agent = DQN(make(), features_dim=64, learning_rate=1e-3, learning_starts=500,
                batch_size=64, buffer_size=10_000, target_update_interval=200,
                exploration_fraction=0.3, seed=0, logger=quiet_logger)
    agent.learn(8_000)
    mean_return, _ = evaluate_policy(agent, make(), n_episodes=20)
    assert mean_return > 0.0, f"DQN failed to learn from pixels (mean_return={mean_return:.3f})"
