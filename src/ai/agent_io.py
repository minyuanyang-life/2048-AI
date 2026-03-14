from pathlib import Path
import json
from typing import Any


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DIR = _PROJECT_ROOT / "artifacts"


def _resolve_file_path(
    agent_name: str,
    path: str | Path | None = None,
) -> Path:
    if path is not None:
        file_path = Path(path)
    else:
        file_path = _DEFAULT_DIR / f"{agent_name}_params.json"

    if not file_path.is_absolute():
        file_path = _PROJECT_ROOT / file_path

    return file_path


def save_params(
    output: dict[str, Any],
    agent_name: str,
    path: str | Path | None = None,
) -> Path:
    file_path = _resolve_file_path(agent_name, path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    return file_path


def load_params(
    agent_name: str,
    path: str | Path | None = None,
) -> dict[str, Any]:
    file_path = _resolve_file_path(agent_name, path)

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return data