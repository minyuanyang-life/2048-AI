from abc import ABC, abstractmethod
from pathlib import Path

from src.ai.evaluator.base_evaluator import BaseEvaluator
from src.core.enums import Direction
from src.core.game import Game

class Agent(ABC):
    """
    Base class for all 2048 agents.

    Responsibilities:
    - Observe the current game state.
    - Choose a next action.
    - Optionally provide action ranking information.

    Notes:
    - Agents should not mutate the input game directly.
    - State transitions should be handled by Game.step().
    """

    def __init__(self, name: str = "agent") -> None:
        self.name = name

    @abstractmethod
    def get_action(self, game: Game) -> Direction:
        """
        Choose the next action from the current game state.

        Input:
            game: the current game instance.

        Output:
            A Direction representing the chosen move.
        """
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def get_action_ranking(self, game: Game) -> list[tuple[Direction, float]]:
        """
        Return a ranking of actions from best to worst.

        Input:
            game: the current game instance.

        Output:
            A list of (direction, score) pairs sorted in descending order.

        Raises:
            NotImplementedError: if the agent does not support action ranking.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support action ranking."
        )

    def supports_action_ranking(self) -> bool:
        return type(self).get_action_ranking is not Agent.get_action_ranking

    def __str__(self) -> str:
        return self.name

class TrainableAgent(Agent):
    def __init__(
        self,
        evaluator: type[BaseEvaluator],
        params,
        name: str = "agent",
    ) -> None:
        super().__init__(name)
        self._evaluator = evaluator(params)
        self._rng = None

    def get_action(self, game: Game) -> Direction:
        return super().get_action(game)

    def record_params(self):
        return self._evaluator.record_params()

    def get_params(self):
        return self._evaluator.get_params()

    def set_params(self, params):
        return self._evaluator.set_params(params)

    def save(self, path: str | Path | None = None) -> Path:
        return self._evaluator.save(path)

    def load(self, path: str | Path | None = None) -> None:
        self._evaluator.load(path)

    def feedback(self) -> None:
        self._evaluator.feedback(self._rng)

    def get_param_vector(self) -> list[float]:
        return self._evaluator.get_param_vector()

    def set_param_vector(self, values: list[float]) -> None:
        self._evaluator.set_param_vector(values)
