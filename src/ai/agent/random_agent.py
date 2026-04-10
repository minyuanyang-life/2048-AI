import random

from src.core.enums import Direction
from src.core.game import Game
from src.ai.agent.agent import Agent


class RandomAgent(Agent):
    def __init__(self, seed: int | None = None) -> None:
        super().__init__("RandomAgent")
        self._rng = random.Random(seed)

    def get_action(self, game: Game) -> Direction:
        dirs = game.board.get_legal_directions()
        if not dirs:
            raise RuntimeError("RandomAgent was called on a terminal game state.")
        return self._rng.choice(dirs)
