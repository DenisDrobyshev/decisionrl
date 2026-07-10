"""Policy + value residual network for AlphaZero."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .games import Game

__all__ = ["AlphaZeroNet"]


class _ResBlock(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class AlphaZeroNet(nn.Module):
    """Shared trunk with a policy head (action logits) and a value head (tanh)."""

    def __init__(self, game: Game, num_resblocks: int = 4, hidden: int = 64) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.ReLU()
        )
        self.res_blocks = nn.ModuleList([_ResBlock(hidden) for _ in range(num_resblocks)])
        board = game.row_count * game.col_count
        self.policy_head = nn.Sequential(
            nn.Conv2d(hidden, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * board, game.action_size),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(hidden, 3, 3, padding=1), nn.BatchNorm2d(3), nn.ReLU(),
            nn.Flatten(), nn.Linear(3 * board, 1), nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        for block in self.res_blocks:
            x = block(x)
        return self.policy_head(x), self.value_head(x)
