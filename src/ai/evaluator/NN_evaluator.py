from pathlib import Path
from random import Random
import time

import torch
import torch.nn as nn

from src.ai.evaluator.base_evaluator import BaseEvaluator
from src.ai.evaluator.config import DEVICE
from src.ai.metrics.profile_store import EpisodeProfile
from src.core.board import Board


class NNEvaluator(BaseEvaluator):
    def __init__(self, params=None) -> None:
        super().__init__("NNEvaluator", params)
        self.device = DEVICE
        self.model = nn.Sequential(
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = nn.MSELoss()
        self.episode_states: list[list[float]] = []
        self.episode_env_rewards: list[float] = []
        self.episode_shaping_rewards: list[float] = []
        self._training_mode = False
        self._profile: EpisodeProfile | None = None

        if isinstance(params, dict):
            model_state = params.get("model_state_dict")
            optimizer_state = params.get("optimizer_state_dict")
            if model_state is not None:
                self.model.load_state_dict(model_state)
            if optimizer_state is not None:
                self.optimizer.load_state_dict(optimizer_state)

        # Default mode is evaluation/test.
        self.set_eval_mode()

    def set_train_mode(self) -> None:
        self._training_mode = True
        self.model.train()

    def set_eval_mode(self) -> None:
        self._training_mode = False
        self.model.eval()

    @property
    def training_mode(self) -> bool:
        return self._training_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def evaluate_board(self, board: Board) -> float:
        start = time.perf_counter()
        features = [
            float(board.get_exponent(r, c))
            for r in range(4)
            for c in range(4)
        ]
        x = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            y = self.forward(x).squeeze()
        value = float(y.item())
        if self._profile is not None:
            self._profile.evaluate_board_calls += 1
            self._profile.evaluate_board_total_s += time.perf_counter() - start
        return value

    def evaluate_boards(self, boards: list[Board]) -> list[float]:
        if not boards:
            return []
        start = time.perf_counter()
        features = [
            [
                float(board.get_exponent(r, c))
                for r in range(4)
                for c in range(4)
            ]
            for board in boards
        ]
        x = torch.tensor(features, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            y = self.forward(x).squeeze(-1)
        values = y.detach().cpu().tolist()
        if isinstance(values, float):
            values = [values]
        values = [float(v) for v in values]
        if self._profile is not None:
            self._profile.evaluate_board_calls += len(values)
            self._profile.evaluate_board_total_s += time.perf_counter() - start
        return values

    def append_state(self, board: Board) -> None:
        features = [
            float(board.get_exponent(r, c))
            for r in range(4)
            for c in range(4)
        ]
        self.episode_states.append(features)

    def append_reward(self, reward: float) -> None:
        # Backward-compatible alias: append to env reward stream.
        self.append_env_reward(reward)

    def append_env_reward(self, reward: float) -> None:
        self.episode_env_rewards.append(float(reward))

    def append_shaping_reward(self, reward: float) -> None:
        self.episode_shaping_rewards.append(float(reward))

    def reset_episode_buffer(self) -> None:
        self.episode_states.clear()
        self.episode_env_rewards.clear()
        self.episode_shaping_rewards.clear()

    def feedback(self, rng: Random) -> None:
        # NN optimization should be triggered by trainer logic.
        return

    def bind_profile(self, profile: EpisodeProfile | None) -> None:
        self._profile = profile

    def train_episode(self, final_score: float) -> float:
        """
        Train once using the cached states from one finished episode.
        Targets are reward-to-go (returns) computed from cached rewards.
        If rewards are missing, fallback target is final_score.
        Returns the scalar training loss.
        """
        start = time.perf_counter()
        if not self.episode_states:
            if self._profile is not None:
                self._profile.train_episode_total_s += time.perf_counter() - start
            return 0.0

        was_training = self._training_mode
        self.set_train_mode()

        x = torch.tensor(
            self.episode_states,
            dtype=torch.float32,
            device=self.device,
        )
        if (
            len(self.episode_env_rewards) == len(self.episode_states)
            and (
                len(self.episode_shaping_rewards) == 0
                or len(self.episode_shaping_rewards) == len(self.episode_states)
            )
        ):
            rewards = list(self.episode_env_rewards)
            if len(self.episode_shaping_rewards) == len(self.episode_states):
                rewards = [a + b for a, b in zip(rewards, self.episode_shaping_rewards)]

            returns = [0.0] * len(rewards)
            running = 0.0
            for i in range(len(rewards) - 1, -1, -1):
                running += rewards[i]
                returns[i] = running
            y = torch.tensor(returns, dtype=torch.float32, device=self.device).unsqueeze(1)
            # Normalize targets per episode to reduce scale drift and variance.
            y_mean = y.mean()
            y_std = y.std(unbiased=False)
            y = (y - y_mean) / (y_std + 1e-6)
        else:
            y = torch.full(
                (x.shape[0], 1),
                float(final_score),
                dtype=torch.float32,
                device=self.device,
            )

        predict = self.forward(x)
        loss = self.criterion(predict, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.reset_episode_buffer()

        if not was_training:
            self.set_eval_mode()

        value = float(loss.item())
        if self._profile is not None:
            self._profile.train_episode_total_s += time.perf_counter() - start
        return value

    def save(self, path: str | Path | None = None) -> Path:
        file_path = self._resolve_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "_training_mode": self._training_mode,
        }
        torch.save(payload, file_path)
        return file_path

    def load(self, path: str | Path | None = None) -> None:
        file_path = self._resolve_path(path)
        checkpoint = torch.load(file_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
        if checkpoint.get("_training_mode", False):
            self.set_train_mode()
        else:
            self.set_eval_mode()

    @staticmethod
    def _resolve_path(path: str | Path | None = None) -> Path:
        project_root = Path(__file__).resolve().parents[3]
        if path is None:
            return project_root / "artifacts" / "nn_evaluator.pt"
        file_path = Path(path)
        if not file_path.is_absolute():
            file_path = project_root / file_path
        return file_path
