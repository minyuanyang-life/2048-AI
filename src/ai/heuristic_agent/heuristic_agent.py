import random
from dataclasses import replace
from math import log2

from src.core.board import Board
from src.core.enums import Direction, MoveStatus
from src.core.game import Game
from src.ai.agent import TrainableAgent
from .heuristic_config import HeuristicWeights


class HeuristicAgent(TrainableAgent):
    def __init__(
        self,
        seed: int | None = None,
        params: HeuristicWeights | None = None,
    ) -> None:
        super().__init__("HeuristicAgent")
        self._rng = random.Random(seed)
        self._weights = replace(params) if params is not None else HeuristicWeights()

    def get_action(self, game: Game) -> Direction:
        dirs = game.board.get_legal_directions()
        if not dirs:
            raise RuntimeError("HeuristicAgent was called on a terminal game state.")
        max_score = float("-inf")
        max_score_direction = []
        for direction in dirs:
            score = self._evaluate_action(game, direction)
            if score > max_score:
                max_score = score
                max_score_direction = [direction]
            elif score == max_score:
                max_score_direction.append(direction)
        return self._rng.choice(max_score_direction)

    def _evaluate_action(self, game: Game, direction: Direction) -> float:
        simulate_board, move_status, score_delta = game.board.simulate_move(direction)
        if move_status == MoveStatus.INVALID_MOVE:
            raise RuntimeError("排除错误方向后依然被尝试")
        feature_empty = self._feature_empty_tiles(simulate_board)
        feature_log2_max = self._feature_log2_max_tile(simulate_board)
        feature_pos_bonus = self._feature_max_tile_position_bonus(simulate_board)
        feature_score_add = self._feature_move_score_add(score_delta)
        return (
            feature_empty * self._weights.empty +
            feature_log2_max * self._weights.log2_max +
            feature_pos_bonus * self._weights.pos_bonus +
            feature_score_add * self._weights.score_add
        )

    def set_params(self, params: HeuristicWeights) -> None:
        self._weights = replace(params)

    def get_params(self) -> HeuristicWeights:
        return replace(self._weights)

    @staticmethod
    def _feature_empty_tiles(board: Board) -> float:
        empty_tiles = board.count_empty_tiles()
        return empty_tiles/16

    @staticmethod
    def _feature_log2_max_tile(board: Board) -> float:
        max_exponent, _ = board.get_max_exponent()
        return max_exponent

    @staticmethod
    def _feature_max_tile_position_bonus(board: Board) -> float:
        _, max_index = board.get_max_exponent()
        endpoint = (0, 3)
        score = 0
        for (x, y) in max_index:
            if x in endpoint and y in endpoint:
                score += 1
            elif x in endpoint or y in endpoint:
                score += 0.5
            else:
                score += 0
        return score/len(max_index)

    @staticmethod
    def _feature_move_score_add(score_add: int) -> float:
        return log2(score_add+1)
