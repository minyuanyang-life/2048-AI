from abc import ABC, abstractmethod

from src.ai.agent.agent import TrainableAgent
from src.core.enums import GameStatus, MoveStatus
from src.core.game import Game


class BaseTrainer(ABC):
    def __init__(self, agent: TrainableAgent) -> None:
        self.agent = agent
        self.name = agent.name + "Trainer"

    @abstractmethod
    def train(self, n: int, base_seed: int = 0) -> float:
        raise NotImplementedError

    def evaluate(self, n: int) -> float:
        score = 0
        for i in range(n):
            game = Game()
            game_status = GameStatus.RUNNING
            while game_status == GameStatus.RUNNING:
                direction = self.agent.get_action(game)
                game_status, move_status, _ = game.step(direction)
                if move_status == MoveStatus.INVALID_MOVE:
                    raise RuntimeError(f"{self.agent.name} produced an invalid move.")
            score += game.score
        return score / n
