from abc import abstractmethod, ABC
from dataclasses import replace
from pathlib import Path
from random import Random


class BaseEvaluator(ABC):
    def __init__(self, name: str = "", params = None) -> None:
        self.name = name
        self._params = params
        self._best_params = self._params

    def record_params(self):
        self._best_params = replace(self._params)

    def set_params(self, params) -> None:
        self._params = replace(params)
        self._best_params = replace(params)

    def get_params(self):
        return replace(self._best_params)

    @abstractmethod
    def save(self, path: str | Path | None = None) -> Path:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str | Path | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def feedback(self, rng: Random) -> None:
        raise NotImplementedError

    def get_param_vector(self) -> list[float]:
        raise NotImplementedError

    def set_param_vector(self, values: list[float]) -> None:
        raise NotImplementedError
