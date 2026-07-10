"""AlphaZero self-play on Tic-Tac-Toe + a README figure.

Trains AlphaZero purely by self-play and, after each iteration, measures its
win / draw / loss rate against a random opponent — the curve shows the agent
learning to never lose. Writes docs/assets/alphazero_learning.png.

Run: python examples/alphazero_demo.py
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from reinforce.alphazero import AlphaZero, TicTacToe, pit, random_player
from reinforce.utils import set_seed

ASSETS = os.path.join(os.path.dirname(__file__), "..", "docs", "assets")
os.makedirs(ASSETS, exist_ok=True)


def main() -> None:
    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    game = TicTacToe()
    agent = AlphaZero(game, n_simulations=60, num_resblocks=3, hidden=48, device=device, seed=0)

    iters = 12
    wins, draws, losses = [], [], []
    rng = np.random.default_rng(100)

    def opponent(s, p):
        return random_player(game, s, p, rng)

    for it in range(iters):
        agent.learn(iterations=1, games_per_iter=30, epochs=4, batch_size=64)
        res = pit(game, lambda s, p: agent.predict(s, p, deterministic=True), opponent, n_games=40)
        wins.append(res["win"])
        draws.append(res["draw"])
        losses.append(res["loss"])
        print(f"iter {it + 1:2d}: {res}")

    x = np.arange(1, iters + 1)
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=110)
    ax.plot(x, wins, "-o", color="#16a34a", label="wins")
    ax.plot(x, draws, "-o", color="#f59e0b", label="draws")
    ax.plot(x, losses, "-o", color="#dc2626", label="losses")
    ax.set_xlabel("self-play iteration")
    ax.set_ylabel("games out of 40 vs random")
    ax.set_title("AlphaZero on Tic-Tac-Toe (self-play, no human data)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "alphazero_learning.png"))
    plt.close(fig)
    print("wrote alphazero_learning.png")


if __name__ == "__main__":
    main()
