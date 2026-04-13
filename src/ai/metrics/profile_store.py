from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass
class EpisodeProfile:
    seed: int
    depth: int
    total_duration_s: float = 0.0
    total_steps: int = 0
    get_action_total_s: float = 0.0
    game_step_total_s: float = 0.0
    train_episode_total_s: float = 0.0
    evaluate_board_calls: int = 0
    evaluate_board_total_s: float = 0.0

    @property
    def evaluate_board_avg_s(self) -> float:
        if self.evaluate_board_calls <= 0:
            return 0.0
        return self.evaluate_board_total_s / self.evaluate_board_calls

    def to_dict(self) -> dict:
        data = asdict(self)
        data["evaluate_board_avg_s"] = self.evaluate_board_avg_s
        return data


class ProfileStore:
    def __init__(self) -> None:
        self.records: list[EpisodeProfile] = []

    def add(self, record: EpisodeProfile) -> None:
        self.records.append(record)

    def to_dict(self) -> dict:
        return {"records": [r.to_dict() for r in self.records]}

    def save_json(self, path: str | Path) -> Path:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return file_path

