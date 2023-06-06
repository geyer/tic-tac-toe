import abc
import os.path
from collections import defaultdict
from copy import deepcopy
from functools import partial
from typing import Sequence, Tuple

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
    chars = {1: "X", -1: "O", 0: " "}
    rows = []
    for row in board:
        row = [chars[i] for i in row]
        rows.append(" | ".join(row))
    separator = "\n" + "-" * 9 + "\n"
    print(separator.join(rows))


class Player:
    @abc.abstractmethod
    def get_move(self, board) -> Move:
        """Returns the next move as (row, col)."""

    def new_game(self):
        pass

    def give_reward(self, _):
        pass


class HumanPlayer(Player):
    def get_move(self, board: Board) -> Move:
        print_board(board)
        words = input("What is your next move (as row, col)? ").split()
        words = list(map(int, words))
        row, col = words
        return row, col


def valid_moves(board: Board):
    moves = [(row, col) for row in range(3)
             for col in range(3) if board[row, col] == 0]
    return moves


def get_symmetries():
    """Returns transforms for board and moves."""

    def reflect(axis, x):
        if len(x) == 2:
            x = list(x)
            x[axis] = 2 - x[axis]
            return tuple(x)
        else:
            return np.flip(x, axis)

    def rotate(turns, x):
        if len(x) == 2:
            for _ in range(turns):
                x = (2-x[1], x[0])
            return x
        else:
            return np.rot90(x, turns)

    def compose(func1, func2):
        def composition(x):
            return func2(func1(x))

        return composition

    symmetries = [
        # Rotations.
        lambda x: x,
        partial(rotate, 1),
        partial(rotate, 2),
        partial(rotate, 3),
        # Reflected rotations.
        partial(reflect, 0),
        partial(reflect, 1),
        compose(partial(reflect, 0), partial(rotate, 1)),
        compose(partial(rotate, 1), partial(reflect, 0)),
    ]
    return symmetries


class BotExhaustiveSearch(Player):
    def find_best_move(self, board: Board, player: int):
        """Returns a best move for the player in the given position.

        Calls func(board, move) for the best move found."""
        score = score_board(board)
        if score is not None:
            return None, score

        moves = valid_moves(board)
        scores = []
        for move in moves:
            mod_board = deepcopy(board)
            mod_board[move] = player
            _, score = self.find_best_move(mod_board, -player)
            scores.append(score)
        # Pick the best move for the current player.
        if player == 1:
            score = max(scores)
        else:
            score = min(scores)
        move = moves[scores.index(score)]
        return move, score

    def get_move(self, board: Board):
        # Infer player number from board.
        player = np.count_nonzero(board) % 2
        player = 1 - 2 * player
        move, _ = self.find_best_move(board, player)
        return move


class BotLookupTable(Player):
    _LOOKUP_TABLE_PATH = "tic_tac_toe_lookup.txt"

    def __init__(self):
        self._transforms = get_symmetries()
        # Load lookup table, or generate and save lookup table.
        best_moves = self.load_table()
        if not best_moves:
            print("Generating lookup table for bot player.")
            best_moves = self.generate_table()
            self.store_table(best_moves)

        print(f"Using lookup table with {len(best_moves)} entries.")
        self._best_moves = best_moves

    def store_table(self, best_moves):
        with open(self._LOOKUP_TABLE_PATH, "w") as f:
            for query, move in best_moves.items():
                f.write(f"{query}: {move}\n")

    def load_table(self):
        if not os.path.exists(self._LOOKUP_TABLE_PATH):
            return None

        def tuple_from_string(text):
            text = text.strip()
            text = text[1:-1]
            return tuple(int(i) for i in text.split(","))

        best_moves = {}
        with open(self._LOOKUP_TABLE_PATH, "r") as f:
            for line in f.readlines():
                query, move = line.split(":")
                query = tuple_from_string(query)
                move = tuple_from_string(move)
                best_moves[query] = move
        return best_moves

    def _key(self, board):
        """Lookup key for board configurations (hashable)."""
        return tuple(board.flatten())

    def generate_table(self):
        best_moves = {}
        best_scores = {}

        def add_best_move(board, move, score):
            # Check permutations first.
            assert board[move] == 0
            query = self._key(board)
            best_moves[query] = move
            best_scores[query] = score

        def dfs(board, player=1):
            score = score_board(board)
            if score is not None:
                return score

            for func in self._transforms:
                query = self._key(func(board))
                if query in best_scores:
                    return best_scores[query]

            moves = valid_moves(board)
            scores = []
            for move in moves:
                mod_board = deepcopy(board)
                mod_board[move] = player
                score = dfs(mod_board, -player)
                scores.append(score)
            score = max(scores)
            if player == -1:
                score = min(scores)
            move = moves[scores.index(score)]
            add_best_move(board, move, score)
            return score

        board = np.zeros((3, 3), dtype=int)
        dfs(board)
        return best_moves

    def get_move(self, board: Board):
        # Retrieve value from lookup table.
        all_moves = [(x, y) for x in range(3) for y in range(3)]
        for func in self._transforms:
            query = self._key(func(board))
            if query in self._best_moves:
                func_move = self._best_moves[query]
                return [m for m in all_moves if func(m) == func_move][0]
        print("Could not find entry in lookup table.")
        return None


class RlPlayer(Player):
    def __init__(self):
        # Initialize value function
        self._value_fn = defaultdict(float)
        self._sequence = []
        self._actions = [(x, y) for x in range(3) for y in range(3)]

    def new_game(self):
        self._sequence = []

    def get_move(self, board: Board):
        player = 1 if np.count_nonzero(board) % 2 == 0 else -1
        # Evaluate value function after each action candidate.
        values = []
        for action in self._actions:
            if board[action] != 0:
                values.append(-1000)
                continue
            modified = deepcopy(board)
            modified[action] = player
            values.append(self._value_fn[tuple(modified.flatten())])
        max_value = max(values)
        indices = [i for i, v in enumerate(values) if v > max_value - 0.01]
        i = np.random.choice(indices)
        action = self._actions[i]
        modified = deepcopy(board)
        modified[action] = player
        self._sequence.append(deepcopy(board))
        self._sequence.append(modified)
        return action

    def give_reward(self, reward: float):
        self._value_fn[tuple(self._sequence[-1].flatten())] = reward
        for board in reversed(self._sequence[:-1]):
            player = 1 if np.count_nonzero(board) % 2 == 0 else -1
            values = []
            for action in self._actions:
                if board[action] != 0:
                    continue
                modified = deepcopy(board)
                modified[action] = player
                values.append(self._value_fn[tuple(modified.flatten())])
            value = sum(values) / len(values)
            self._value_fn[tuple(board.flatten())] = value


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
            move = self._players[next_player].get_move(board)
            self.apply_move(board, next_player, move)
            score = score_board(board)
            next_player = (next_player + 1) % 2
        if score == 0:
            print("Game ends in tie!")
            self._result_counts[0] += 1
            pass
        else:
            winner = 2 - next_player
            self._result_counts[2 * next_player - 1] += 1
            print(f"Player {winner} wins.")
        print_board(board)

    def run_with_reward(self):
        for player in self._players:
            player.new_game()
        board = np.zeros((3, 3), dtype=int)
        last_player = None
        next_player = 0
        score = None
        while score is None:
            move = self._players[next_player].get_move(board)
            try:
                self.apply_move(board, next_player, move)
                score = score_board(board)
                last_player,  next_player = next_player, (next_player + 1) % 2
            except:
                print(f"invalid move {move}")
                print_board(board)
                self._result_counts['invalid'] += 1
                self._players[next_player].give_reward(-100)
                return
        self._players[last_player].give_reward(abs(score))
        self._players[next_player].give_reward(-abs(score))
        if score == 0:
            self._result_counts[0] += 1
        else:
            self._result_counts[last_player] += 1


if __name__ == '__main__':

    TicTacToe(
        [
            HumanPlayer(),
            BotLookupTable(),
        ]
    ).run()
