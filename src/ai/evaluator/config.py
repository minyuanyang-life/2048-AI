from dataclasses import dataclass
from typing import List, Tuple

import torch

try:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"


@dataclass
class HeuristicWeights:
    empty: float = 1.0
    log2_max: float = 1.0
    snake_monotonicity: float = 1.0

    def to_list(self) -> list[float]:
        return [self.empty, self.log2_max, self.snake_monotonicity]

    @classmethod
    def from_list(cls, values: list[float]) -> "HeuristicWeights":
        if len(values) != 3:
            raise ValueError(f"Expected 3 values, got {len(values)}")
        return cls(
            empty=values[0],
            log2_max=values[1],
            snake_monotonicity=values[2]
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "empty": self.empty,
            "log2_max": self.log2_max,
            "snake_monotonicity": self.snake_monotonicity
        }

    @classmethod
    def from_dict(cls, values: dict[str, float]) -> "HeuristicWeights":
        expected_keys = {"empty", "log2_max", "snake_monotonicity"}
        actual_keys = set(values.keys())

        if actual_keys != expected_keys:
            raise ValueError(
                f"Expected keys {expected_keys}, got {actual_keys}"
            )
        return cls(
            empty=values["empty"],
            log2_max=values["log2_max"],
            snake_monotonicity=values["snake_monotonicity"]
        )

@dataclass
class HeuristicTrainerConfig:
    num_iterations: int = 5000
    num_eval_games: int = 100
    sample_sigma: float = 0.1

@dataclass
class ExpectimaxConfig:
    depth: int = 2
    spawn_two_prob: float = 0.9
    spawn_four_prob: float = 0.1
    tie_break_eps: float = 1e-9

SNAKE_TEMPLATES: List[List[Tuple[int, int]]] = [
    [  # top-left, row-first
        (0, 0), (0, 1), (0, 2), (0, 3),
        (1, 3), (1, 2), (1, 1), (1, 0),
        (2, 0), (2, 1), (2, 2), (2, 3),
        (3, 3), (3, 2), (3, 1), (3, 0),
    ],
    [  # top-left, col-first
        (0, 0), (1, 0), (2, 0), (3, 0),
        (3, 1), (2, 1), (1, 1), (0, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (3, 3), (2, 3), (1, 3), (0, 3),
    ],
    [  # top-right, row-first
        (0, 3), (0, 2), (0, 1), (0, 0),
        (1, 0), (1, 1), (1, 2), (1, 3),
        (2, 3), (2, 2), (2, 1), (2, 0),
        (3, 0), (3, 1), (3, 2), (3, 3),
    ],
    [  # top-right, col-first
        (0, 3), (1, 3), (2, 3), (3, 3),
        (3, 2), (2, 2), (1, 2), (0, 2),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (3, 0), (2, 0), (1, 0), (0, 0),
    ],
    [  # bottom-left, row-first
        (3, 0), (3, 1), (3, 2), (3, 3),
        (2, 3), (2, 2), (2, 1), (2, 0),
        (1, 0), (1, 1), (1, 2), (1, 3),
        (0, 3), (0, 2), (0, 1), (0, 0),
    ],
    [  # bottom-left, col-first
        (3, 0), (2, 0), (1, 0), (0, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (3, 2), (2, 2), (1, 2), (0, 2),
        (0, 3), (1, 3), (2, 3), (3, 3),
    ],
    [  # bottom-right, row-first
        (3, 3), (3, 2), (3, 1), (3, 0),
        (2, 0), (2, 1), (2, 2), (2, 3),
        (1, 3), (1, 2), (1, 1), (1, 0),
        (0, 0), (0, 1), (0, 2), (0, 3),
    ],
    [  # bottom-right, col-first
        (3, 3), (2, 3), (1, 3), (0, 3),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (3, 1), (2, 1), (1, 1), (0, 1),
        (0, 0), (1, 0), (2, 0), (3, 0),
    ],
]
