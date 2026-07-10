"""AlphaZero: MCTS + self-play for two-player perfect-information games.

    from reinforce.alphazero import AlphaZero, TicTacToe
    agent = AlphaZero(TicTacToe(), n_simulations=60, seed=0)
    agent.learn(iterations=10, games_per_iter=30)
    action = agent.predict(state, player=1)     # MCTS-backed move
"""

from .alphazero import AlphaZero, pit, random_player
from .games import Connect4, Game, TicTacToe
from .mcts import MCTS, Node
from .network import AlphaZeroNet

__all__ = [
    "Game",
    "TicTacToe",
    "Connect4",
    "AlphaZeroNet",
    "MCTS",
    "Node",
    "AlphaZero",
    "pit",
    "random_player",
]
