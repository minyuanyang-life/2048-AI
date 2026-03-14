import unittest
from src.core.move import MoveEngine
from src.core.enums import Direction, GameState
from src.core.board import Board


class TestSlideAndMerge(unittest.TestCase):
    def test_all_zero(self):
        new_line, score = MoveEngine._slide_and_merge([0, 0, 0, 0])
        self.assertEqual(new_line, [0, 0, 0, 0])
        self.assertEqual(score, 0)

    def test_slide_only(self):
        new_line, score = MoveEngine._slide_and_merge([0, 2, 0, 4])
        self.assertEqual(new_line, [2, 4, 0, 0])
        self.assertEqual(score, 0)

    def test_simple_merge(self):
        new_line, score = MoveEngine._slide_and_merge([2, 0, 2, 0])
        self.assertEqual(new_line, [4, 0, 0, 0])
        self.assertEqual(score, 4)

    def test_merge_chain_not_allowed(self):
        # 同一步里，一个块不能连合并
        new_line, score = MoveEngine._slide_and_merge([2, 2, 2, 0])
        self.assertEqual(new_line, [4, 2, 0, 0])
        self.assertEqual(score, 4)

    def test_double_pair_merge(self):
        new_line, score = MoveEngine._slide_and_merge([2, 2, 2, 2])
        self.assertEqual(new_line, [4, 4, 0, 0])
        self.assertEqual(score, 8)

    def test_merge_after_slide(self):
        new_line, score = MoveEngine._slide_and_merge([0, 2, 2, 4])
        self.assertEqual(new_line, [4, 4, 0, 0])
        self.assertEqual(score, 4)

    def test_no_merge(self):
        new_line, score = MoveEngine._slide_and_merge([2, 4, 2, 4])
        self.assertEqual(new_line, [2, 4, 2, 4])
        self.assertEqual(score, 0)


class TestApplyAllDirections(unittest.TestCase):
    def make_board(self):
        board = Board()
        board._grid = [
            [2, 0, 2, 2],
            [0, 0, 0, 2],
            [2, 2, 4, 4],
            [0, 2, 0, 2],
        ]
        return board

    def test_apply_left(self):
        board = self.make_board()
        result, new_board, score= MoveEngine.apply(board, Direction.LEFT)
        self.assertEqual(result, GameState.SUCCESS)
        self.assertEqual(new_board._grid, [
            [4, 2, 0, 0],
            [2, 0, 0, 0],
            [4, 8, 0, 0],
            [4, 0, 0, 0],
        ])
        self.assertEqual(score, 20)

    def test_apply_right(self):
        board = self.make_board()
        result, new_board, score = MoveEngine.apply(board, Direction.RIGHT)
        self.assertEqual(result, GameState.SUCCESS)
        self.assertEqual(new_board._grid, [
            [0, 0, 2, 4],
            [0, 0, 0, 2],
            [0, 0, 4, 8],
            [0, 0, 0, 4],
        ])
        self.assertEqual(score, 20)

    def test_apply_up(self):
        board = self.make_board()
        result, new_board, score = MoveEngine.apply(board, Direction.UP)
        self.assertEqual(result, GameState.SUCCESS)
        self.assertEqual(new_board._grid, [
            [4, 4, 2, 4],
            [0, 0, 4, 4],
            [0, 0, 0, 2],
            [0, 0, 0, 0],
        ])
        self.assertEqual(score, 12)

    def test_apply_down(self):
        board = self.make_board()
        result, new_board, score = MoveEngine.apply(board, Direction.DOWN)
        self.assertEqual(result, GameState.SUCCESS)
        self.assertEqual(new_board._grid, [
            [0, 0, 0, 0],
            [0, 0, 0, 4],
            [0, 0, 2, 4],
            [4, 4, 4, 2],
        ])
        self.assertEqual(score, 12)

class TestApplyNone(unittest.TestCase):
    def make_board(self):
        board = Board()
        board._grid = [
            [0, 0, 0, 2],
            [0, 0, 0, 2],
            [2, 4, 2, 4],
            [0, 0, 0, 2],
        ]
        return board

    def test_apply(self):
        board = self.make_board()
        result, new_board, score= MoveEngine.apply(board, Direction.RIGHT)
        self.assertEqual(result, GameState.FAIL)
        self.assertEqual(new_board._grid, [
            [0, 0, 0, 2],
            [0, 0, 0, 2],
            [2, 4, 2, 4],
            [0, 0, 0, 2],
        ])
        self.assertEqual(score, 0)


if __name__ == "__main__":
    unittest.main()
