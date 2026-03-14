from dataclasses import replace
from math import log2

from src.core.board import Board
from src.ai.evaluator.heuristic_config import HeuristicWeights, SNAKE_TEMPLATES


class HeuristicEvaluator:
    def __init__(self, weights: HeuristicWeights | None = None) -> None:
        self._weights = replace(weights) if weights is not None else HeuristicWeights()

    def evaluate_board(self, board: Board) -> float:
        feature_empty = self._feature_empty_tiles(board)
        feature_log2_max = self._feature_log2_max_tile(board)
        feature_snake_monotonicity = self._feature_snake_monotonicity(board)

        return (
            feature_empty * self._weights.empty
            + feature_log2_max * self._weights.log2_max
            + feature_snake_monotonicity * self._weights.snake_monotonicity
        )

    def set_weights(self, weights: HeuristicWeights) -> None:
        self._weights = replace(weights)

    def get_weights(self) -> HeuristicWeights:
        return replace(self._weights)

    def _feature_snake_monotonicity(self, board: Board) -> float:
        weights = [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        best_score = float("-inf")

        for template in SNAKE_TEMPLATES:
            values = [board.grid[r][c] for r, c in template]
            score = self._score_single_snake(values, weights)
            best_score = max(best_score, score)

        return best_score

    @staticmethod
    def _score_single_snake(values: list[int], weights: list[int]) -> float:
        monotonicity_penalty = 0.0
        hole_penalty = 0.0

        non_zero_with_pos = [(i, v) for i, v in enumerate(values) if v != 0]

        for k in range(len(non_zero_with_pos) - 1):
            idx1, v1 = non_zero_with_pos[k]
            idx2, v2 = non_zero_with_pos[k + 1]

            if v2 > v1:
                diff = v2 - v1
                monotonicity_penalty += diff * weights[idx1]

        seen_zero = False
        for i, v in enumerate(values):
            if v == 0:
                seen_zero = True
            elif seen_zero:
                hole_penalty += weights[i]

        monotonicity_weight = 1.0
        hole_weight = 0.5

        total_penalty = (
            monotonicity_weight * monotonicity_penalty
            + hole_weight * hole_penalty
        )

        return -total_penalty

    @staticmethod
    def _feature_empty_tiles(board: Board) -> float:
        empty_tiles = board.count_empty_tiles()
        return empty_tiles ** 2 / 16

    @staticmethod
    def _feature_log2_max_tile(board: Board) -> float:
        max_exponent, _ = board.get_max_exponent()
        return max_exponent

    @staticmethod
    def _feature_max_tile_position_bonus(board: Board) -> float:
        _, max_index = board.get_max_exponent()
        endpoint = (0, 3)
        score = 0
        for x, y in max_index:
            if x in endpoint and y in endpoint:
                score += 1
            elif x in endpoint or y in endpoint:
                score += 0.5
        return score / len(max_index)

    @staticmethod
    def _feature_move_score_add(score_add: int) -> float:
        return log2(score_add + 1)