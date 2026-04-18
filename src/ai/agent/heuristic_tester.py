import json
import os
import random
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from tqdm import tqdm

from src.ai.agent.expectimax_agent import ExpectimaxAgent
from src.ai.evaluator.heuristic_evaluator import HeuristicEvaluator
from src.ai.metrics.profile_store import EpisodeProfile
from src.core.enums import GameStatus, MoveStatus
from src.core.game import Game

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


class HeuristicTester:
    """
    Stable evaluator for Heuristic-based expectimax agent.
    """

    def __init__(
        self,
        seed: int = 0,
        depth: int = 2,
    ) -> None:
        self.seed = int(seed)
        self._rng = random.Random(self.seed)
        self.agent = ExpectimaxAgent(evaluator=HeuristicEvaluator, seed=self.seed)
        self.agent.config.depth = depth

    @staticmethod
    def _set_global_seed(seed: int) -> None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _run_one_game(self, seed: int) -> dict:
        self._set_global_seed(seed)
        if hasattr(self.agent, "_rng") and self.agent._rng is not None:
            self.agent._rng.seed(seed + 1)

        evaluator = self.agent.evaluator
        profile = EpisodeProfile(seed=seed, depth=int(self.agent.config.depth))
        if hasattr(evaluator, "bind_profile"):
            evaluator.bind_profile(profile)

        game = Game()
        game_status = GameStatus.RUNNING
        t0 = time.perf_counter()

        while game_status == GameStatus.RUNNING:
            t_action = time.perf_counter()
            direction = self.agent.get_action(game)
            profile.get_action_total_s += time.perf_counter() - t_action

            t_step = time.perf_counter()
            game_status, move_status, _ = game.step(direction)
            profile.game_step_total_s += time.perf_counter() - t_step
            if move_status == MoveStatus.INVALID_MOVE:
                raise RuntimeError(f"{self.agent.name} produced an invalid move.")

        profile.total_duration_s = time.perf_counter() - t0
        profile.total_steps = int(game.steps)
        max_exp, _ = game.board.get_max_exponent()
        if hasattr(evaluator, "bind_profile"):
            evaluator.bind_profile(None)

        return {
            "seed": seed,
            "score": float(game.score),
            "steps": profile.total_steps,
            "max_exponent": int(max_exp),
            "reached_512": bool(max_exp >= 9),
            "reached_1024": bool(max_exp >= 10),
            "reached_2048": bool(max_exp >= 11),
            "reached_4096": bool(max_exp >= 12),
            "duration_s": float(profile.total_duration_s),
            "get_action_total_s": float(profile.get_action_total_s),
            "game_step_total_s": float(profile.game_step_total_s),
            "train_episode_total_s": float(profile.train_episode_total_s),
            "evaluate_board_calls": int(profile.evaluate_board_calls),
            "evaluate_board_total_s": float(profile.evaluate_board_total_s),
            "evaluate_board_avg_s": float(profile.evaluate_board_avg_s),
        }

    def run(
        self,
        num_games: int = 20,
        base_seed: int = 0,
        output_path: str | Path | None = None,
        output_image_path: str | Path | None = None,
    ) -> dict:
        if num_games <= 0:
            raise ValueError("num_games must be > 0")

        games: list[dict] = []
        for i in tqdm(range(num_games), desc=f"Heuristic Test(d={self.agent.config.depth})"):
            games.append(self._run_one_game(base_seed + i))

        scores = [g["score"] for g in games]
        steps = [g["steps"] for g in games]
        durations = [g["duration_s"] for g in games]
        get_action_totals = [g["get_action_total_s"] for g in games]
        game_step_totals = [g["game_step_total_s"] for g in games]
        eval_calls = [g["evaluate_board_calls"] for g in games]
        eval_totals = [g["evaluate_board_total_s"] for g in games]
        eval_avgs = [g["evaluate_board_avg_s"] for g in games]
        hit_512 = sum(1 for g in games if g["reached_512"])
        hit_1024 = sum(1 for g in games if g["reached_1024"])
        hit_2048 = sum(1 for g in games if g["reached_2048"])
        hit_4096 = sum(1 for g in games if g["reached_4096"])

        summary = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "depth": int(self.agent.config.depth),
            "num_games": int(num_games),
            "base_seed": int(base_seed),
            "score_mean": float(statistics.fmean(scores)),
            "score_std": float(statistics.pstdev(scores)) if len(scores) > 1 else 0.0,
            "score_min": float(min(scores)),
            "score_max": float(max(scores)),
            "steps_mean": float(statistics.fmean(steps)),
            "duration_mean_s": float(statistics.fmean(durations)),
            "duration_total_s": float(sum(durations)),
            "get_action_mean_s": float(statistics.fmean(get_action_totals)),
            "get_action_total_s": float(sum(get_action_totals)),
            "game_step_mean_s": float(statistics.fmean(game_step_totals)),
            "game_step_total_s": float(sum(game_step_totals)),
            "evaluate_board_calls_total": int(sum(eval_calls)),
            "evaluate_board_calls_mean": float(statistics.fmean(eval_calls)),
            "evaluate_board_total_s": float(sum(eval_totals)),
            "evaluate_board_avg_s_mean": float(statistics.fmean(eval_avgs)),
            "reach_512_count": int(hit_512),
            "reach_1024_count": int(hit_1024),
            "reach_2048_count": int(hit_2048),
            "reach_4096_count": int(hit_4096),
        }
        report = {"summary": summary, "games": games}

        if output_path is not None:
            file_path = Path(output_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with file_path.open("w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

        if output_image_path is not None:
            image_path = self._plot_report(report, output_image_path)
            if image_path is not None:
                self._open_image(image_path)

        return report

    @staticmethod
    def _plot_report(report: dict, output_image_path: str | Path) -> Path | None:
        if plt is None:
            return None

        games = report["games"]
        idx = list(range(1, len(games) + 1))
        scores = [g["score"] for g in games]
        max_exps = [g["max_exponent"] for g in games]
        durations = [g["duration_s"] for g in games]
        action_s = [g["get_action_total_s"] for g in games]
        step_s = [g["game_step_total_s"] for g in games]
        eval_calls = [g["evaluate_board_calls"] for g in games]
        hit_512 = [1 if g["reached_512"] else 0 for g in games]
        hit_1024 = [1 if g["reached_1024"] else 0 for g in games]
        hit_2048 = [1 if g["reached_2048"] else 0 for g in games]
        hit_4096 = [1 if g["reached_4096"] else 0 for g in games]
        cum_512_count = []
        cum_1024_count = []
        cum_2048_count = []
        cum_4096_count = []
        s512 = 0
        s1024 = 0
        s2048 = 0
        s4096 = 0
        for v512, v1024, v2048, v4096 in zip(hit_512, hit_1024, hit_2048, hit_4096):
            s512 += v512
            s1024 += v1024
            s2048 += v2048
            s4096 += v4096
            cum_512_count.append(s512)
            cum_1024_count.append(s1024)
            cum_2048_count.append(s2048)
            cum_4096_count.append(s4096)

        fig, axes = plt.subplots(3, 2, figsize=(13, 11))
        fig.suptitle("Heuristic Test Report", fontsize=14)

        ax = axes[0, 0]
        ax.plot(idx, scores, marker="o", linewidth=1)
        ax.set_title("Score by Game")
        ax.set_xlabel("Game Index")
        ax.set_ylabel("Score")

        ax = axes[0, 1]
        ax.hist(scores, bins=min(10, max(3, len(scores))), edgecolor="black")
        ax.set_title("Score Distribution")
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")

        ax = axes[1, 0]
        ax.plot(idx, durations, marker="o", linewidth=1, label="total")
        ax.plot(idx, action_s, marker="x", linewidth=1, label="get_action")
        ax.plot(idx, step_s, marker=".", linewidth=1, label="game_step")
        ax.set_title("Duration by Game")
        ax.set_xlabel("Game Index")
        ax.set_ylabel("Seconds")
        ax.legend()

        ax = axes[1, 1]
        ax.plot(idx, cum_512_count, marker="o", linewidth=1, label=">=512")
        ax.plot(idx, cum_1024_count, marker="o", linewidth=1, label=">=1024")
        ax.plot(idx, cum_2048_count, marker="o", linewidth=1, label=">=2048")
        ax.plot(idx, cum_4096_count, marker="o", linewidth=1, label=">=4096")
        ax.set_title("Cumulative Hit Counts")
        ax.set_xlabel("Game Index")
        ax.set_ylabel("Count")
        ax.legend()

        ax = axes[2, 0]
        ax.plot(idx, max_exps, marker="o", linewidth=1)
        ax.set_title("Max Exponent by Game")
        ax.set_xlabel("Game Index")
        ax.set_ylabel("Exponent")

        ax = axes[2, 1]
        ax.plot(idx, eval_calls, marker="o", linewidth=1)
        ax.set_title("evaluate_board Calls by Game")
        ax.set_xlabel("Game Index")
        ax.set_ylabel("Calls")

        fig.tight_layout()
        out_path = Path(output_image_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        return out_path

    @staticmethod
    def _open_image(path: Path) -> None:
        try:
            if os.name == "nt":
                os.startfile(str(path))  # type: ignore[attr-defined]
            elif os.name == "posix":
                opener = "open" if "darwin" in os.sys.platform else "xdg-open"
                subprocess.Popen([opener, str(path)])
        except Exception:
            pass


def main() -> None:
    tester = HeuristicTester(seed=0, depth=2)
    report = tester.run(
        num_games=20,
        base_seed=0,
        output_path="artifacts/heuristic_test_report_depth2.json",
        output_image_path="artifacts/heuristic_test_report_depth2.png",
    )
    s = report["summary"]
    print(
        f"depth={s['depth']} games={s['num_games']} "
        f"mean={s['score_mean']:.1f} std={s['score_std']:.1f} "
        f"hit512={s['reach_512_count']} "
        f"hit1024={s['reach_1024_count']} "
        f"hit2048={s['reach_2048_count']} "
        f"hit4096={s['reach_4096_count']} "
        f"plot=artifacts/heuristic_test_report_depth2.png"
    )


if __name__ == "__main__":
    main()
