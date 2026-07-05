"""Convolutional feature extractor for image observations.

A compact, size-agnostic CNN (conv -> ReLU -> max-pool blocks) that flattens to a
dense feature vector. The flattened dimension is computed from a dummy forward,
so it adapts to any ``(C, H, W)`` input. Use :class:`ImageQNetwork` to plug it
into DQN, or reuse :class:`CNNFeatureExtractor` in your own heads.
"""

from __future__ import annotations

from typing import Sequence, Tuple, Type

import torch
import torch.nn as nn

from .mlp import layer_init

__all__ = ["CNNFeatureExtractor", "ImageQNetwork", "is_image_space"]


def is_image_space(space) -> bool:
    """True for observation spaces shaped like a channel-first image ``(C, H, W)``."""
    return space.shape is not None and len(space.shape) == 3


class CNNFeatureExtractor(nn.Module):
    def __init__(
        self,
        obs_shape: Tuple[int, int, int],
        features_dim: int = 256,
        channels: Sequence[int] = (16, 32),
        activation: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        c, h, w = obs_shape
        layers: list = []
        in_c = c
        for out_c in channels:
            layers += [nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1), activation(), nn.MaxPool2d(2)]
            in_c = out_c
        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, c, h, w)).flatten(1).shape[1]
        self.head = nn.Sequential(nn.Flatten(), layer_init(nn.Linear(n_flat, features_dim)), activation())
        self.features_dim = int(features_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(obs))


class ImageQNetwork(nn.Module):
    """CNN feature extractor + linear Q-head for discrete actions."""

    def __init__(
        self,
        obs_shape: Tuple[int, int, int],
        n_actions: int,
        features_dim: int = 256,
        channels: Sequence[int] = (16, 32),
        activation: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.features = CNNFeatureExtractor(obs_shape, features_dim, channels, activation)
        self.head = layer_init(nn.Linear(features_dim, n_actions), gain=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(obs))
