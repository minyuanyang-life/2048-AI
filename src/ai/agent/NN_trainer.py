import random
import time
import torch
from tqdm import tqdm

from src.ai.agent.base_trainer import BaseTrainer
from src.ai.agent.expectimax_agent import ExpectimaxAgent
from src.ai.evaluator.NN_evaluator import NNEvaluator
from src.ai.evaluator.heuristic_evaluator import HeuristicEvaluator
from src.ai.metrics.profile_store import EpisodeProfile, ProfileStore
from src.core.enums import GameStatus, MoveStatus
from src.core.game import Game


class NNTrainer(BaseTrainer):
    def __init__(self, seed: int | None = None) -> None:
        super().__init__(ExpectimaxAgent(evaluator=NNEvaluator, seed=seed))
        self.seed = 0 if seed is None else int(seed)
        self._rng = random.Random(self.seed)
        self._shape_evaluator = HeuristicEvaluator()
        self._apply_seed(self.seed)

    @staticmethod
    def _set_global_seed(seed: int) -> None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _apply_seed(self, seed: int) -> None:
        self._set_global_seed(seed)
        self._rng.seed(seed + 1)
        if hasattr(self.agent, "_rng") and self.agent._rng is not None:
            self.agent._rng.seed(seed + 2)

    def _play_one_game(self, seed: int | None = None) -> float:
        if seed is not None:
            self._apply_seed(seed)
        game = Game()
        game_status = GameStatus.RUNNING
        while game_status == GameStatus.RUNNING:
            direction = self.agent.get_action(game)
            game_status, move_status, _ = game.step(direction)
            if move_status == MoveStatus.INVALID_MOVE:
                raise RuntimeError(f"{self.agent.name} produced an invalid move.")
        return float(game.score)

    def _compute_shaping_reward(self, game: Game) -> float:
        """
        Small shaping reward:
        - encourage max-tile corner strategy
        - encourage snake-like monotonic layout
        """
        max_exp, max_positions = game.board.get_max_exponent()
        if max_exp < 0:
            return 0.0

        corners = ((0, 0), (0, 3), (3, 0), (3, 3))
        corner_bonus = 0.05 * max_exp
        distance_penalty = 0.03 * max_exp

        # Best corner distance among all max-tile positions.
        min_dist = min(
            abs(r - cr) + abs(c - cc)
            for r, c in max_positions
            for cr, cc in corners
        )
        on_corner = any((r, c) in corners for r, c in max_positions)

        reward = -distance_penalty * min_dist
        if on_corner:
            reward += corner_bonus

        snake_score = self._shape_evaluator.snake_monotonicity_score(game.board)
        snake_reward = 0.01 * snake_score

        reward += snake_reward

        # Keep shaping bounded so env reward remains dominant.
        return max(-2.0, min(2.0, reward))

    def _evaluate_mean_score(self, num_games: int, base_seed: int, episode_idx: int) -> float:
        if num_games <= 0:
            return 0.0
        evaluator = self.agent.evaluator
        evaluator.set_eval_mode()
        total = 0.0
        eval_seed_base = base_seed + 1_000_000 + episode_idx * 1_000
        for i in range(num_games):
            total += self._play_one_game(seed=eval_seed_base + i)
            if self.pbar is not None:
                self.pbar.update(1)
        evaluator.set_train_mode()
        return total / num_games

    def profile_depth_runs(
            self,
            seed: int,
            depths: tuple[int, ...] = (1, 2),
            run_train_episode: bool = True,
            output_path: str | None = None,
    ) -> ProfileStore:
        """
        Same-seed one-episode profiling for multiple depths.
        Records:
        1) episode total duration
        2) total steps
        3) cumulative get_action duration
        4) cumulative game.step duration
        5) train_episode duration
        6) NNEvaluator.evaluate_board call count / total / avg duration
        """
        store = ProfileStore()
        evaluator = self.agent.evaluator
        evaluator.set_train_mode()

        for depth in depths:
            self.agent.config.depth = depth
            self._apply_seed(seed)

            record = EpisodeProfile(seed=seed, depth=depth)
            evaluator.bind_profile(record)
            evaluator.reset_episode_buffer()

            game = Game()
            game_status = GameStatus.RUNNING
            total_start = time.perf_counter()

            while game_status == GameStatus.RUNNING:
                evaluator.append_state(game.board)

                action_start = time.perf_counter()
                direction = self.agent.get_action(game)
                record.get_action_total_s += time.perf_counter() - action_start

                step_start = time.perf_counter()
                game_status, move_status, info = game.step(direction)
                record.game_step_total_s += time.perf_counter() - step_start
                if move_status == MoveStatus.INVALID_MOVE:
                    raise RuntimeError(f"{self.agent.name} produced an invalid move.")

                evaluator.append_env_reward(float(info.get("score_delta", 0.0)))
                evaluator.append_shaping_reward(self._compute_shaping_reward(game))

            record.total_duration_s = time.perf_counter() - total_start
            record.total_steps = game.steps

            if run_train_episode:
                evaluator.train_episode(float(game.score))
            else:
                evaluator.reset_episode_buffer()

            evaluator.bind_profile(None)
            store.add(record)

        if output_path is not None:
            store.save_json(output_path)

        evaluator.set_eval_mode()
        return store

    def pretrain_with_teacher(
            self,
            num_games: int = 100,
            max_steps_per_game: int = 300,
            epochs: int = 3,
            batch_size: int = 256,
            target_scale: float = 10.0,
            base_seed: int = 100_000,
    ) -> float:
        """
        Supervised warmup:
        - collect states from random-play games
        - use HeuristicEvaluator as teacher target
        - train NNEvaluator by MSE
        """
        if num_games <= 0 or epochs <= 0:
            return 0.0

        evaluator = self.agent.evaluator
        teacher = HeuristicEvaluator()
        try:
            teacher.load()
        except FileNotFoundError:
            # If no teacher checkpoint exists, use default heuristic weights.
            pass

        states: list[list[float]] = []
        targets: list[float] = []

        for game_idx in tqdm(range(num_games), desc="Collect Teacher Data"):
            self._apply_seed(base_seed + game_idx)
            game = Game()
            game_status = GameStatus.RUNNING
            steps = 0

            while game_status == GameStatus.RUNNING and steps < max_steps_per_game:
                states.append([
                    float(game.board.get_exponent(r, c))
                    for r in range(4)
                    for c in range(4)
                ])
                targets.append(float(teacher.evaluate_board(game.board)) / target_scale)

                legal_dirs = game.board.get_legal_directions()
                if not legal_dirs:
                    break
                direction = self._rng.choice(legal_dirs)
                game_status, move_status, _ = game.step(direction)
                if move_status == MoveStatus.INVALID_MOVE:
                    raise RuntimeError("Random policy produced invalid move unexpectedly.")
                steps += 1

        if not states:
            return 0.0

        x_all = torch.tensor(states, dtype=torch.float32, device=evaluator.device)
        y_all = torch.tensor(targets, dtype=torch.float32, device=evaluator.device).unsqueeze(1)

        evaluator.set_train_mode()
        n_samples = x_all.shape[0]
        last_loss = 0.0
        for _ in tqdm(range(epochs), desc="Teacher Pretrain"):
            perm = torch.randperm(n_samples, device=evaluator.device)
            for start in range(0, n_samples, batch_size):
                idx = perm[start:start + batch_size]
                xb = x_all[idx]
                yb = y_all[idx]

                pred = evaluator.forward(xb)
                loss = evaluator.criterion(pred, yb)
                evaluator.optimizer.zero_grad()
                loss.backward()
                evaluator.optimizer.step()
                last_loss = float(loss.item())

        evaluator.set_eval_mode()
        self.agent.save()
        tqdm.write(f"teacher pretrain done: samples={n_samples}, final_loss={last_loss:.4f}")
        return last_loss

    def train(
            self,
            n: int,
            base_seed: int = 0,
            eval_interval: int = 20,
            eval_games: int = 5,
    ) -> float:
        # Keep compatibility with existing signature: n is number of episodes.
        if n <= 0:
            return 0.0

        try:
            self.agent.load()
        except FileNotFoundError:
            # First training run may not have a checkpoint yet.
            pass

        evaluator = self.agent.evaluator
        evaluator.set_train_mode()

        total_score = 0.0
        total_loss = 0.0
        best_eval_score = float("-inf")
        latest_eval = 0.0

        eval_rounds = 0
        if eval_interval > 0 and eval_games > 0:
            eval_rounds = n // eval_interval
        total_units = n + eval_rounds * eval_games

        self.pbar = tqdm(total=total_units, desc="NN Training")
        for episode_idx in range(n):
            self._apply_seed(base_seed + episode_idx)
            game = Game()
            game_status = GameStatus.RUNNING

            evaluator.reset_episode_buffer()

            while game_status == GameStatus.RUNNING:
                evaluator.append_state(game.board)

                direction = self.agent.get_action(game)
                game_status, move_status, info = game.step(direction)
                if move_status == MoveStatus.INVALID_MOVE:
                    raise RuntimeError(f"{self.agent.name} produced an invalid move.")

                evaluator.append_env_reward(float(info.get("score_delta", 0.0)))
                evaluator.append_shaping_reward(self._compute_shaping_reward(game))

            total_score += float(game.score)
            loss = evaluator.train_episode(float(game.score))
            total_loss += loss
            self.pbar.update(1)

            if (
                eval_interval > 0
                and eval_games > 0
                and (episode_idx + 1) % eval_interval == 0
            ):
                mean_eval = self._evaluate_mean_score(eval_games, base_seed, episode_idx)
                latest_eval = mean_eval
                if mean_eval > best_eval_score:
                    best_eval_score = mean_eval
                    self.agent.save()
                    tqdm.write(
                        f"new best eval mean = {best_eval_score:.1f} "
                        f"({eval_games} games), checkpoint saved"
                    )

            self.pbar.set_postfix(
                avg_score=f"{total_score / (episode_idx + 1):.1f}",
                mean_eval=f"{latest_eval:.1f}",
                loss=f"{loss:.4f}",
            )

        evaluator.set_eval_mode()
        self.agent.save()

        avg_score = total_score / n
        avg_loss = total_loss / n
        tqdm.write(f"NN training done: avg_score={avg_score:.1f}, avg_loss={avg_loss:.4f}")
        return avg_score


def main() -> None:
    trainer = NNTrainer(seed=0)
    enable_teacher_pretrain = False
    if enable_teacher_pretrain:
        trainer.pretrain_with_teacher(
            num_games=100,
            max_steps_per_game=300,
            epochs=3,
            batch_size=256,
            target_scale=10.0,
            base_seed=100_000,
        )
    # trainer.train(n=500, base_seed=0, eval_interval=10, eval_games=10)
    trainer.train(n=120, base_seed=0, eval_interval=12, eval_games=5)


if __name__ == "__main__":
    main()
