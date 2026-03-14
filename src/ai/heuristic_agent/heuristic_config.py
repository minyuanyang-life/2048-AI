from dataclasses import dataclass

@dataclass
class HeuristicWeights:
    empty: float = 4.695681645390938
    log2_max: float = 2.8410676043151595
    pos_bonus: float = 3.1284674766948743
    score_add: float = 1.0378088347508525

    def to_list(self) -> list[float]:
        return [self.empty, self.log2_max, self.pos_bonus, self.score_add]

    @classmethod
    def from_list(cls, values: list[float]) -> "HeuristicWeights":
        if len(values) != 4:
            raise ValueError(f"Expected 4 values, got {len(values)}")
        return cls(
            empty=values[0],
            log2_max=values[1],
            pos_bonus=values[2],
            score_add=values[3],
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "empty": self.empty,
            "log2_max": self.log2_max,
            "pos_bonus": self.pos_bonus,
            "score_add": self.score_add,
        }

    @classmethod
    def from_dict(cls, values: dict[str, float]) -> "HeuristicWeights":
        expected_keys = {"empty", "log2_max", "pos_bonus", "score_add"}
        actual_keys = set(values.keys())

        if actual_keys != expected_keys:
            raise ValueError(
                f"Expected keys {expected_keys}, got {actual_keys}"
            )
        return cls(
            empty=values["empty"],
            log2_max=values["log2_max"],
            pos_bonus=values["pos_bonus"],
            score_add=values["score_add"],
        )

@dataclass
class HeuristicTrainerConfig:
    num_iterations: int = 500
    num_eval_games: int = 1000
    sample_sigma: float = 0.1