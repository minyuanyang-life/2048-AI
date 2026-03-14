from collections import defaultdict
from dataclasses import dataclass

from src.core.enums import Direction


@dataclass
class HeuristicWeights:
    empty: float = 0.5269
    log2_max: float = 0.9679
    pos_bonus: float = 0.6071
    score_add: float = 2.1161

    def __str__(self):
        return f"{self.empty:.4f} {self.log2_max:.4f} {self.pos_bonus:.4f} {self.score_add:.4f}"

@dataclass
class RLParameters:
    epsilon: float = 0.1
    gamma: float = 0.99
    epsilon_decay: float = 0.99

