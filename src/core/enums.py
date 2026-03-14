from enum import Enum, IntEnum, auto


class Direction(IntEnum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3

    @classmethod
    def all(cls) -> tuple["Direction", ...]:
        return cls.LEFT, cls.RIGHT, cls.UP, cls.DOWN

    def __str__(self) -> str:
        return self.name

class MoveStatus(Enum):
    INVALID_MOVE = auto()
    MOVED = auto()

class GameStatus(Enum):
    RUNNING = auto()
    GAME_OVER = auto()
    WIN = auto()