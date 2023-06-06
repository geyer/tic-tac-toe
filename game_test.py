import game
import unittest
import numpy as np


class TestSymmetries(unittest.TestCase):

    def test_symmetries_on_board(self):
        """Tests that the symmetries leave rows of the board intact."""
        lines = [[0, 1, 2], [3, 4, 5], [6, 7, 8],
                 [0, 3, 6], [1, 4, 7], [2, 5, 8]]
        lines = lines + [x[::-1] for x in lines]

        symmetries = game.get_symmetries()
        for func in symmetries:
            board = np.arange(9, dtype=int).reshape(3, 3)
            modified = func(board.copy())

            self.assertEqual(set(modified.flatten()), set(board.flatten()))

            for row in modified:
                self.assertIn(list(row), lines)

    def test_symmetries(self):
        """Tests that symmetries on board and moves match."""
        board = np.arange(9).reshape(3, 3)
        moves = [(i, j) for i in range(3) for j in range(3)]
        for func in game.get_symmetries():
            modified = func(board.copy())
            values = [board[move] for move in moves]
            func_values = [modified[func(move)] for move in moves]
            self.assertSequenceEqual(values, func_values)


if __name__ == '__main__':
    unittest.main()
