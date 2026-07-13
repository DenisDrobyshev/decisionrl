"""Monte-Carlo Tree Search with a neural-network prior (PUCT), for AlphaZero."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

from .games import Game

__all__ = ["MCTS", "Node"]


class Node:
    def __init__(self, game: Game, c_puct: float, state, parent: Optional["Node"] = None,
                 action_taken: Optional[int] = None, prior: float = 0.0, visit_count: int = 0) -> None:
        self.game = game
        self.c_puct = c_puct
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        self.children: list = []
        self.visit_count = visit_count
        self.value_sum = 0.0

    def is_fully_expanded(self) -> bool:
        return len(self.children) > 0

    def select(self) -> "Node":
        return max(self.children, key=self._ucb)

    def _ucb(self, child: "Node") -> float:
        # child value is from the opponent's perspective -> invert to ours
        q = 0.0 if child.visit_count == 0 else 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q + self.c_puct * child.prior * math.sqrt(self.visit_count) / (1 + child.visit_count)

    def expand(self, policy: np.ndarray) -> None:
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.game.get_next_state(self.state.copy(), action, player=1)
                child_state = self.game.change_perspective(child_state, player=-1)
                self.children.append(
                    Node(self.game, self.c_puct, child_state, parent=self, action_taken=action, prior=float(prob))
                )

    def backpropagate(self, value: float) -> None:
        self.value_sum += value
        self.visit_count += 1
        if self.parent is not None:
            self.parent.backpropagate(self.game.get_opponent_value(value))


class MCTS:
    def __init__(self, game: Game, model, n_simulations: int = 100, c_puct: float = 1.5,
                 dirichlet_alpha: float = 0.3, dirichlet_eps: float = 0.25, device="cpu") -> None:
        self.game = game
        self.model = model
        self.n_simulations = int(n_simulations)
        self.c_puct = float(c_puct)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.dirichlet_eps = float(dirichlet_eps)
        self.device = device
        self.rng = np.random.default_rng()

    def _policy_value(self, state):
        enc = torch.as_tensor(self.game.get_encoded_state(state), dtype=torch.float32, device=self.device)
        logits, value = self.model(enc.unsqueeze(0))
        policy = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        return policy, float(value.item())

    @torch.no_grad()
    def search(self, state, add_noise: bool = True) -> np.ndarray:
        self.model.eval()
        root = Node(self.game, self.c_puct, state, visit_count=1)
        policy, _ = self._policy_value(state)
        valid = self.game.get_valid_moves(state)
        if add_noise:
            noise = self.rng.dirichlet([self.dirichlet_alpha] * self.game.action_size)
            policy = (1 - self.dirichlet_eps) * policy + self.dirichlet_eps * noise
        policy *= valid
        policy /= policy.sum() if policy.sum() > 0 else 1.0
        root.expand(policy)

        for _ in range(self.n_simulations):
            node = root
            while node.is_fully_expanded():
                node = node.select()
            value, terminated = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)
            if not terminated:
                policy, value = self._policy_value(node.state)
                valid = self.game.get_valid_moves(node.state)
                policy *= valid
                policy /= policy.sum() if policy.sum() > 0 else 1.0
                node.expand(policy)
            node.backpropagate(value)

        action_probs = np.zeros(self.game.action_size, dtype=np.float32)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        total = action_probs.sum()
        return action_probs / total if total > 0 else action_probs
