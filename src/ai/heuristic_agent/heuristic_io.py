from pathlib import Path
import json

from .heuristic_config import HeuristicWeights


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DIR = _PROJECT_ROOT / "artifacts"
_DEFAULT_FILE = _DEFAULT_DIR / "heuristic_weights.json"


def save_weights(
    weights: HeuristicWeights,
    path: str | Path | None = None,
) -> Path:
    file_path = Path(path) if path is not None else _DEFAULT_FILE
    if not file_path.is_absolute():
        file_path = _PROJECT_ROOT / file_path

    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as f:
        json.dump(weights.to_dict(), f, indent=4)

    return file_path


def load_weights(path: str | Path | None = None) -> HeuristicWeights:
    file_path = Path(path) if path is not None else _DEFAULT_FILE
    if not file_path.is_absolute():
        file_path = _PROJECT_ROOT / file_path

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return HeuristicWeights.from_dict(data)