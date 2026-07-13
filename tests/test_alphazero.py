"""Tests for AlphaZero: games, network, MCTS and self-play learning."""

import numpy as np
import pytest

from decisionrl.alphazero import (
    AlphaZero,
    AlphaZeroNet,
    Connect4,
    TicTacToe,
    pit,
    random_player,
)
from decisionrl.alphazero.mcts import MCTS


def test_tictactoe_mechanics():
    game = TicTacToe()
    state = game.get_initial_state()
    for a, p in [(0, 1), (3, -1), (1, 1), (4, -1)]:
        state = game.get_next_state(state, a, p)
    assert not game.check_win(state, 1)
    state = game.get_next_state(state, 2, 1)  # completes the top row for +1
    value, terminated = game.get_value_and_terminated(state, 2)
    assert game.check_win(state, 2) and terminated and value == 1.0
    assert game.get_valid_moves(state).sum() == 4


def test_connect4_mechanics():
    game = Connect4()
    state = game.get_initial_state()
    for _ in range(3):  # three pieces stacked in column 0
        state = game.get_next_state(state, 0, 1)
    assert not game.check_win(state, 0)
    state = game.get_next_state(state, 0, 1)  # fourth -> vertical win
    assert game.check_win(state, 0)
    assert game.get_valid_moves(state).shape == (7,)


def test_network_output_shapes():
    import torch

    game = Connect4()
    net = AlphaZeroNet(game, num_resblocks=2, hidden=16)
    x = torch.zeros(4, 3, game.row_count, game.col_count)
    policy, value = net(x)
    assert policy.shape == (4, game.action_size)
    assert value.shape == (4, 1)


def test_mcts_returns_valid_distribution():
    game = TicTacToe()
    net = AlphaZeroNet(game, num_resblocks=2, hidden=16).eval()
    mcts = MCTS(game, net, n_simulations=30, device="cpu")
    state = game.get_next_state(game.get_initial_state(), 4, 1)  # one move played
    probs = mcts.search(game.change_perspective(state, -1), add_noise=False)
    assert probs.shape == (9,)
    assert np.isclose(probs.sum(), 1.0)
    # no probability mass on the occupied centre cell
    assert probs[4] == 0.0


def test_mcts_finds_immediate_win():
    game = TicTacToe()
    state = game.get_initial_state()
    for a, p in [(0, 1), (3, -1), (1, 1), (4, -1)]:  # +1 threatens to win at cell 2
        state = game.get_next_state(state, a, p)
    net = AlphaZeroNet(game, num_resblocks=2, hidden=32).eval()
    mcts = MCTS(game, net, n_simulations=120, device="cpu")
    probs = mcts.search(state, add_noise=False)
    assert int(np.argmax(probs)) == 2  # search discovers the winning move


def test_alphazero_save_load(tmp_path):
    game = TicTacToe()
    agent = AlphaZero(game, n_simulations=25, num_resblocks=2, hidden=16, device="cpu", seed=0)
    state = game.get_next_state(game.get_initial_state(), 0, 1)
    a1 = agent.predict(state, player=-1, deterministic=True)
    path = str(tmp_path / "az.pt")
    agent.save(path)
    loaded = AlphaZero.load(path, game=TicTacToe(), n_simulations=25, num_resblocks=2, hidden=16)
    assert loaded.predict(state, player=-1, deterministic=True) == a1


@pytest.mark.slow
def test_alphazero_beats_random_on_tictactoe():
    game = TicTacToe()
    agent = AlphaZero(game, n_simulations=40, num_resblocks=2, hidden=32, device="cpu", seed=0)
    agent.learn(iterations=5, games_per_iter=16, epochs=4, batch_size=64)

    rng = np.random.default_rng(0)
    n = 24
    res = pit(
        game,
        lambda s, p: agent.predict(s, p, deterministic=True),
        lambda s, p: random_player(game, s, p, rng),
        n_games=n,
        seed=1,
    )
    assert res["win"] >= 0.5 * n
    assert res["loss"] <= 0.25 * n
