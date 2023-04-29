from typing import Sequence, Tuple
import abc
from copy import deepcopy

import numpy as np

Board = np.ndarray
Move = Tuple[int, int]


def score_board(board: Board):
    """Returns the score of the current board (or None)."""

    def score_line(line):
        line = np.array(line, dtype=int)
        if np.all(line == 1):
            return 1
        elif np.all(line == -1):
            return -1
        return None

    for row in range(3):
        score = score_line(board[row, :])
        if score:
            return score
    for col in range(3):
        score = score_line(board[:, col])
        if score:
            return score
    diag = [board[i, i] for i in range(3)]
    score = score_line(diag)
    if score:
        return score
    diag = [board[i, 2 - i] for i in range(3)]
    score = score_line(diag)
    if score:
        return score

    if np.all(board != 0):
        return 0

    return None


def print_board(board: Board):
    print(board)


class Player:
    @abc.abstractmethod
    def get_move(self, board) -> Move:
        """Returns the next move as (row, col)."""


class HumanPlayer(Player):
    def get_move(self, board: Board) -> Move:
        words = input("What is your next move (as row, col)? ").split()
        words = list(map(int, words))
        row, col = words
        return row, col


class BotExhaustiveSearch(Player):
    def __init__(self, player):
        self._player = 1 - 2 * player

    def valid_moves(self, board: Board):
        moves = [
            (row, col) for row in range(3) for col in range(3) if board[row, col] == 0
        ]
        return moves

    def best_move(self, board: Board, player):
        score = score_board(board)
        if score is not None:
            return None, score

        moves = self.valid_moves(board)
        scores = []
        for move in moves:
            mod_board = deepcopy(board)
            mod_board[move] = player
            _, score = self.best_move(mod_board, -player)
            scores.append(score)
        if player == 1:
            score = max(scores)
        else:
            score = min(scores)
        return moves[scores.index(score)], score

    def get_move(self, board: Board):
        move, _ = self.best_move(board, self._player)
        return move


class TicTacToe:
    def __init__(self, players: Sequence[Player]):
        self._players = tuple(players)

    def apply_move(self, board: Board, player: int, move: Move):
        # Map player index [0, 1] to values [1, -1].
        player = 1 - 2 * player
        move = np.array(move, dtype=int)
        if np.any(move < 0) or np.any(move > 2):
            raise ValueError(f"Move {move} is out of bounds.")
        move = tuple(move)
        if board[move] != 0:
            raise ValueError(f"Position is already occupied")
        board[move] = player

    def run(self):
        board = np.zeros((3, 3), dtype=int)
        next_player = 0
        score = None
        while score is None:
            print(f"Player {next_player + 1} moves next:")
            print_board(board)
            move = self._players[next_player].get_move(board)
            self.apply_move(board, next_player, move)
            score = score_board(board)
            next_player = (next_player + 1) % 2
        if score == 0:
            print("Game ends in tie!")
        else:
            winner = 2 - next_player
            print(f"Player {winner} wins.")
        print_board(board)


TicTacToe(
    [
        BotExhaustiveSearch(0),
        BotExhaustiveSearch(1),
    ]
).run()
