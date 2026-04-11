import random

from src.ai.agent.agent import TrainableAgent
from src.ai.evaluator.config import ExpectimaxConfig, HeuristicWeights
from src.ai.evaluator.heuristic_evaluator import HeuristicEvaluator
from src.core.enums import Direction, MoveStatus
from src.core.game import Game


class ExpectimaxAgent(TrainableAgent):
    """
    A shallow Expectimax agent for 2048.

    Strategy:
    1. Try each legal player move.
    2. After the move, enumerate all possible random spawns (2 or 4).
    3. Evaluate the expected value of the resulting states.
    4. Choose the move with the highest expected value.

    Notes:
    - Leaf evaluation is delegated to the selected evaluator.
    - board.move(direction) must be deterministic here:
      move/merge only, without spawning a random tile.
    """

    def __init__(
        self,
        config: ExpectimaxConfig | None = None,
        params: HeuristicWeights | None = None,
        evaluator: HeuristicEvaluator = HeuristicEvaluator,
        seed: int | None = None,
    ) -> None:
        super().__init__("ExpectimaxAgent", params, evaluator)
        self.config = config or ExpectimaxConfig()
        self._rng = random.Random(seed)

    def get_action(self, game: Game) -> Direction:
        ranking = self.get_action_ranking(game)
        if not ranking:
            raise RuntimeError("ExpectimaxAgent was called on a terminal game state.")

        best_score = ranking[0][1]
        eps = self.config.tie_break_eps
        best_directions = [
            direction
            for direction, score in ranking
            if abs(score - best_score) <= eps
        ]
        return self._rng.choice(best_directions)

    def get_action_ranking(self, game: Game) -> list[tuple[Direction, float]]:
        dirs = game.board.get_legal_directions()
        if not dirs:
            raise RuntimeError("ExpectimaxAgent was called on a terminal game state.")

        depth = max(1, self.config.depth)

        ranking: list[tuple[Direction, float]] = []
        for direction in dirs:
            score = self._evaluate_action(game, direction, depth)
            ranking.append((direction, score))

        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking

    def _evaluate_action(
        self,
        game: Game,
        direction: Direction,
        depth: int
    ) -> float:
        next_game = game.clone()
        move_status, _ = next_game.board.move(direction)

        if move_status != MoveStatus.MOVED:
            return float("-inf")

        return self._chance_value(next_game, depth - 1)

    def _chance_value(self, game: Game, depth: int) -> float:
        empty_positions = self._get_empty_positions(game)
        if not empty_positions:
            if depth <= 0:
                return self._leaf_score(game)
            return self._max_value(game, depth)

        expected_score = 0.0
        p_cell = 1.0 / len(empty_positions)

        for row, col in empty_positions:
            game_2 = game.clone()
            game_2.board.set_value(row, col, 2)
            if depth <= 0:
                score_2 = self._leaf_score(game_2)
            else:
                score_2 = self._max_value(game_2, depth)
            expected_score += p_cell * self.config.spawn_two_prob * score_2

            game_4 = game.clone()
            game_4.board.set_value(row, col, 4)
            if depth <= 0:
                score_4 = self._leaf_score(game_4)
            else:
                score_4 = self._max_value(game_4, depth)
            expected_score += p_cell * self.config.spawn_four_prob * score_4

        return expected_score

    def _max_value(self, game: Game, depth: int) -> float:
        if depth <= 0:
            return self._leaf_score(game)

        dirs = game.board.get_legal_directions()
        if not dirs:
            return -1e9

        best_score = float("-inf")
        for direction in dirs:
            score = self._evaluate_action(game, direction, depth)
            best_score = max(best_score, score)

        return best_score

    def _leaf_score(self, game: Game) -> float:
        if not game.board.can_move():
            return -1e9
        return self._evaluator.evaluate_board(game.board)

    @staticmethod
    def _get_empty_positions(game: Game) -> list[tuple[int, int]]:
        positions: list[tuple[int, int]] = []
        for row in range(4):
            for col in range(4):
                if game.board.grid[row][col] == 0:
                    positions.append((row, col))
        return positions