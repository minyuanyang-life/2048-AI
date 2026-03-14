from abc import ABC, abstractmethod
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
    @abstractmethod
    def get_params(self):
        raise NotImplementedError

    @abstractmethod
    def set_params(self, params) -> None:
        raise NotImplementedError

    @abstractmethod
    def save(self, ):
        raise NotImplementedError

    @abstractmethod
    def load(self):
        raise NotImplementedError