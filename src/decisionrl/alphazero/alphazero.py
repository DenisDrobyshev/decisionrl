"""AlphaZero: self-play reinforcement learning with MCTS (Silver et al., 2017).

Learns purely from self-play — MCTS acts as a policy-improvement operator over the
network, and the network is trained to imitate the improved (visit-count) policy
and to predict the game outcome. No human data, no reward shaping.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..utils.torch_utils import get_device
from .games import Game
from .mcts import MCTS
from .network import AlphaZeroNet

__all__ = ["AlphaZero", "pit", "random_player"]


class AlphaZero:
    def __init__(
        self,
        game: Game,
        n_simulations: int = 100,
        c_puct: float = 1.5,
        num_resblocks: int = 4,
        hidden: int = 64,
        learning_rate: float = 1e-3,
        temperature: float = 1.0,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
        device: str = "auto",
        seed: Optional[int] = None,
    ) -> None:
        self.game = game
        self.device = get_device(device)
        self.temperature = float(temperature)
        self.model = AlphaZeroNet(game, num_resblocks, hidden).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.mcts = MCTS(game, self.model, n_simulations, c_puct, dirichlet_alpha, dirichlet_eps, self.device)
        self.rng = np.random.default_rng(seed)
        self.mcts.rng = self.rng
        if seed is not None:
            torch.manual_seed(seed)
        self.iterations_done = 0

    # -- self-play ---------------------------------------------------------
    def self_play(self) -> list:
        memory, state, player = [], self.game.get_initial_state(), 1
        while True:
            neutral = self.game.change_perspective(state, player)
            action_probs = self.mcts.search(neutral, add_noise=True)
            memory.append((neutral, action_probs, player))

            temp = action_probs ** (1.0 / self.temperature)
            temp = temp / temp.sum() if temp.sum() > 0 else action_probs
            action = int(self.rng.choice(self.game.action_size, p=temp))

            state = self.game.get_next_state(state, action, player)
            value, terminated = self.game.get_value_and_terminated(state, action)
            if terminated:
                return [
                    (self.game.get_encoded_state(s), p, value if pl == player else -value)
                    for s, p, pl in memory
                ]
            player = self.game.get_opponent(player)

    def _train_epoch(self, memory, batch_size):
        self.rng.shuffle(memory)
        losses = []
        for i in range(0, len(memory), batch_size):
            batch = memory[i: i + batch_size]
            states = torch.as_tensor(np.array([b[0] for b in batch]), dtype=torch.float32, device=self.device)
            target_p = torch.as_tensor(np.array([b[1] for b in batch]), dtype=torch.float32, device=self.device)
            target_v = torch.as_tensor(np.array([b[2] for b in batch]), dtype=torch.float32, device=self.device).unsqueeze(1)

            out_p, out_v = self.model(states)
            policy_loss = -(target_p * F.log_softmax(out_p, dim=1)).sum(dim=1).mean()
            value_loss = F.mse_loss(out_v, target_v)
            loss = policy_loss + value_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.item()))
        return float(np.mean(losses)) if losses else 0.0

    def learn(self, iterations: int = 20, games_per_iter: int = 20, epochs: int = 4,
              batch_size: int = 64, callback=None, log_interval: int = 1):
        """Alternate self-play data generation and network training."""
        for it in range(iterations):
            self.model.eval()
            memory: list = []
            for _ in range(games_per_iter):
                memory += self.self_play()
            self.model.train()
            loss = 0.0
            for _ in range(epochs):
                loss = self._train_epoch(memory, batch_size)
            self.iterations_done += 1
            if callback is not None:
                callback(it, loss)
        return self

    # -- inference ---------------------------------------------------------
    def predict(self, state, player: int = 1, deterministic: bool = True) -> int:
        neutral = self.game.change_perspective(state, player)
        probs = self.mcts.search(neutral, add_noise=not deterministic)
        if deterministic:
            return int(np.argmax(probs))
        return int(self.rng.choice(self.game.action_size, p=probs))

    def save(self, path: str) -> None:
        torch.save({"model": self.model.state_dict()}, path)

    @classmethod
    def load(cls, path: str, game: Game = None, device: str = "auto", **kwargs) -> "AlphaZero":
        agent = cls(game, device=device, **kwargs)
        checkpoint = torch.load(path, map_location=get_device(device), weights_only=False)
        agent.model.load_state_dict(checkpoint["model"])
        return agent


def random_player(game: Game, state, player: int, rng) -> int:
    valid = game.get_valid_moves(game.change_perspective(state, player))
    return int(rng.choice(np.flatnonzero(valid)))


def pit(game: Game, player1: Callable, player2: Callable, n_games: int = 100,
        seed: Optional[int] = None) -> dict:
    """Play ``n_games`` (alternating who starts). Returns win/draw/loss for player1.

    ``player1``/``player2`` are callables ``(state, player) -> action``. ``seed`` is
    accepted for API symmetry; determinism comes from the player callables.
    """
    results = {"win": 0, "draw": 0, "loss": 0}
    players = {1: player1, -1: player2}
    for g in range(n_games):
        first = 1 if g % 2 == 0 else -1  # alternate the starting side
        state, player = game.get_initial_state(), first
        while True:
            action = players[player](state, player)
            state = game.get_next_state(state, action, player)
            value, terminated = game.get_value_and_terminated(state, action)
            if terminated:
                if value == 0:
                    results["draw"] += 1
                elif player == 1:
                    results["win"] += 1
                else:
                    results["loss"] += 1
                break
            player = game.get_opponent(player)
    return results
