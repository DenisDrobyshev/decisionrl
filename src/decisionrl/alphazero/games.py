"""Two-player, perfect-information board games for AlphaZero self-play.

Games use the standard AlphaZero *canonical* convention: the state is always
represented from the current player's perspective (their pieces are ``+1``), so a
single network can play both sides. State is a NumPy board; the interface is
functional (states are passed around) which keeps MCTS simple.
"""

from __future__ import annotations

import numpy as np

__all__ = ["Game", "TicTacToe", "Connect4"]


class Game:
    """Interface for a two-player, zero-sum, perfect-information game."""

    name: str
    row_count: int
    col_count: int
    action_size: int

    def get_initial_state(self) -> np.ndarray:
        return np.zeros((self.row_count, self.col_count), dtype=np.float32)

    def get_next_state(self, state: np.ndarray, action: int, player: int) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def get_valid_moves(self, state: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def check_win(self, state: np.ndarray, action: int) -> bool:  # pragma: no cover
        raise NotImplementedError

    def get_value_and_terminated(self, state: np.ndarray, action: int):
        if self.check_win(state, action):
            return 1.0, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0.0, True
        return 0.0, False

    @staticmethod
    def get_opponent(player: int) -> int:
        return -player

    @staticmethod
    def get_opponent_value(value: float) -> float:
        return -value

    @staticmethod
    def change_perspective(state: np.ndarray, player: int) -> np.ndarray:
        return state * player

    def get_encoded_state(self, state: np.ndarray) -> np.ndarray:
        """Three planes: opponent pieces (-1), empties (0), current pieces (+1)."""
        encoded = np.stack((state == -1, state == 0, state == 1)).astype(np.float32)
        return encoded


class TicTacToe(Game):
    name = "tictactoe"
    row_count = 3
    col_count = 3
    action_size = 9

    def get_next_state(self, state, action, player):
        state = state.copy()
        state[action // self.col_count, action % self.col_count] = player
        return state

    def get_valid_moves(self, state):
        return (state.reshape(-1) == 0).astype(np.uint8)

    def check_win(self, state, action):
        if action is None:
            return False
        row, col = action // self.col_count, action % self.col_count
        player = state[row, col]
        if player == 0:
            return False
        return bool(
            np.sum(state[row, :]) == player * self.col_count
            or np.sum(state[:, col]) == player * self.row_count
            or np.sum(np.diag(state)) == player * self.row_count
            or np.sum(np.diag(np.fliplr(state))) == player * self.row_count
        )


class Connect4(Game):
    name = "connect4"
    row_count = 6
    col_count = 7
    action_size = 7
    in_a_row = 4

    def get_next_state(self, state, action, player):
        state = state.copy()
        row = np.max(np.where(state[:, action] == 0))  # lowest empty cell in the column
        state[row, action] = player
        return state

    def get_valid_moves(self, state):
        return (state[0] == 0).astype(np.uint8)

    def check_win(self, state, action):
        if action is None:
            return False
        row = np.min(np.where(state[:, action] != 0))  # last piece dropped in this column
        col = action
        player = state[row, col]
        if player == 0:
            return False

        def count(dr, dc):
            n = 0
            r, c = row + dr, col + dc
            while 0 <= r < self.row_count and 0 <= c < self.col_count and state[r, c] == player:
                n += 1
                r, c = r + dr, c + dc
            return n

        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            if 1 + count(dr, dc) + count(-dr, -dc) >= self.in_a_row:
                return True
        return False
