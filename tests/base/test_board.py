from src.core.board import Board
import unittest

class TestBoard(unittest.TestCase):
    def make_board(self, grid):
        board = Board()
        board._grid = grid
        return board

    def test_full_true(self):
        grid = [
            [2, 2, 2, 2],
            [2, 4, 2, 2],
            [2, 2, 4, 4],
            [4, 2, 2, 2],
        ]
        board = self.make_board(grid)
        result = board.is_board_full()
        self.assertEqual(result, True)

    def test_full_false(self):
        grid = [
            [2, 0, 2, 2],
            [0, 0, 0, 2],
            [2, 2, 4, 4],
            [0, 2, 0, 2],
        ]
        board = self.make_board(grid)
        result = board.is_board_full()
        self.assertEqual(result, False)