import random
from dataclasses import replace
from pathlib import Path

from src.ai.agent_io import save_params, load_params
from src.core.enums import Direction, MoveStatus
from src.core.game import Game
from src.ai.agent.agent import TrainableAgent
from src.ai.evaluator.heuristic_config import HeuristicWeights
from src.ai.evaluator.heuristic_evaluator import HeuristicEvaluator


class HeuristicAgent(TrainableAgent):
    def __init__(
        self,
        seed: int | None = None,
        params: HeuristicWeights | None = None,
    ) -> None:
        super().__init__("HeuristicAgent")
        self._rng = random.Random(seed)
        self._evaluator = HeuristicEvaluator(params)

    def get_action_ranking(self, game: Game) -> list[tuple[Direction, float]]:
        dirs = game.board.get_legal_directions()
        if not dirs:
            raise RuntimeError("HeuristicAgent was called on a terminal game state.")

        ranking = []
        for direction in dirs:
            score = self._evaluate_action(game, direction)
            ranking.append((direction, score))

        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking

    def get_action(self, game: Game) -> Direction:
        ranking = self.get_action_ranking(game)
        if not ranking:
            raise RuntimeError("HeuristicAgent was called on a terminal game state.")

        max_score = ranking[0][1]
        best_directions = [
            direction
            for direction, score in ranking
            if score == max_score
        ]
        return self._rng.choice(best_directions)

    def _evaluate_action(self, game: Game, direction: Direction) -> float:
        simulate_board, move_status, _ = game.board.simulate_move(direction)
        if move_status == MoveStatus.INVALID_MOVE:
            raise RuntimeError("排除错误方向后依然被尝试")
        return self._evaluator.evaluate_board(simulate_board)

    def set_params(self, params: HeuristicWeights) -> None:
        self._evaluator.set_weights(params)

    def get_params(self) -> HeuristicWeights:
        return self._evaluator.get_weights()

    def save(self, path: str | Path | None = None) -> Path:
        return save_params(self.get_params().to_dict(), "heuristic", path)

    def load(self, path: str | Path | None = None) -> None:
        data = load_params("heuristic", path)
        self.set_params(HeuristicWeights.from_dict(data))