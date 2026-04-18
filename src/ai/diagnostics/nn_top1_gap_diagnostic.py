import json
import math
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
from src.ai.evaluator.NN_evaluator import NNEvaluator
from src.core.enums import GameStatus, MoveStatus
from src.core.game import Game

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0:
        return float(min(values))
    if q >= 1:
        return float(max(values))
    sorted_values = sorted(values)
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return float(sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac)


class NNTop1GapDiagnostic:
    """
    State-level decision confidence diagnostic:
    top1-gap = score(best action) - score(second best action)
    """

    def __init__(
        self,
        seed: int = 0,
        depth: int = 2,
        model_path: str | Path | None = None,
    ) -> None:
        self.seed = int(seed)
        self.agent = ExpectimaxAgent(evaluator=NNEvaluator, seed=self.seed)
        self.agent.config.depth = depth
        try:
            self.agent.load(model_path)
        except FileNotFoundError:
            pass

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

        game = Game()
        game_status = GameStatus.RUNNING
        t0 = time.perf_counter()
        gaps: list[float] = []

        while game_status == GameStatus.RUNNING:
            ranking = self.agent.get_action_ranking(game)
            if not ranking:
                raise RuntimeError("ExpectimaxAgent was called on a terminal game state.")

            top1 = float(ranking[0][1])
            top2 = float(ranking[1][1]) if len(ranking) >= 2 else float("-inf")
            gap = top1 - top2 if math.isfinite(top2) else float("inf")
            if math.isfinite(gap):
                gaps.append(float(gap))

            eps = self.agent.config.tie_break_eps
            best_dirs = [d for d, s in ranking if abs(float(s) - top1) <= eps]
            direction = self.agent._rng.choice(best_dirs)

            game_status, move_status, _ = game.step(direction)
            if move_status == MoveStatus.INVALID_MOVE:
                raise RuntimeError(f"{self.agent.name} produced an invalid move.")

        duration_s = time.perf_counter() - t0
        max_exp, _ = game.board.get_max_exponent()
        return {
            "seed": int(seed),
            "score": float(game.score),
            "steps": int(game.steps),
            "max_exponent": int(max_exp),
            "duration_s": float(duration_s),
            "state_count": int(len(gaps)),
            "top1_gaps": gaps,
            "top1_gap_mean": float(statistics.fmean(gaps)) if gaps else 0.0,
            "top1_gap_median": float(statistics.median(gaps)) if gaps else 0.0,
        }

    def run(
        self,
        num_games: int = 20,
        base_seed: int = 0,
        small_gap_threshold: float = 0.1,
        tiny_gap_threshold: float = 0.01,
        output_path: str | Path | None = None,
        output_image_path: str | Path | None = None,
    ) -> dict:
        if num_games <= 0:
            raise ValueError("num_games must be > 0")

        games: list[dict] = []
        for i in tqdm(range(num_games), desc=f"NN Gap Diagnostic(d={self.agent.config.depth})"):
            games.append(self._run_one_game(base_seed + i))

        all_gaps = [g for game in games for g in game["top1_gaps"]]
        total_states = len(all_gaps)
        small_count = sum(1 for g in all_gaps if g <= small_gap_threshold)
        tiny_count = sum(1 for g in all_gaps if g <= tiny_gap_threshold)

        summary = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "depth": int(self.agent.config.depth),
            "num_games": int(num_games),
            "base_seed": int(base_seed),
            "state_count_total": int(total_states),
            "small_gap_threshold": float(small_gap_threshold),
            "tiny_gap_threshold": float(tiny_gap_threshold),
            "small_gap_count": int(small_count),
            "tiny_gap_count": int(tiny_count),
            "small_gap_ratio": float(small_count / total_states) if total_states > 0 else 0.0,
            "tiny_gap_ratio": float(tiny_count / total_states) if total_states > 0 else 0.0,
            "top1_gap_mean": float(statistics.fmean(all_gaps)) if all_gaps else 0.0,
            "top1_gap_median": float(statistics.median(all_gaps)) if all_gaps else 0.0,
            "top1_gap_p10": float(_quantile(all_gaps, 0.10)) if all_gaps else 0.0,
            "top1_gap_p25": float(_quantile(all_gaps, 0.25)) if all_gaps else 0.0,
            "top1_gap_p75": float(_quantile(all_gaps, 0.75)) if all_gaps else 0.0,
            "top1_gap_p90": float(_quantile(all_gaps, 0.90)) if all_gaps else 0.0,
        }
        report = {"summary": summary, "games": games}

        if output_path is not None:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", encoding="utf-8") as f:
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
        summary = report["summary"]
        idx = list(range(1, len(games) + 1))
        per_game_gap_mean = [g["top1_gap_mean"] for g in games]
        per_game_gap_median = [g["top1_gap_median"] for g in games]
        per_game_small_ratio = []
        all_gaps = []
        for g in games:
            gaps = g["top1_gaps"]
            all_gaps.extend(gaps)
            if gaps:
                per_game_small_ratio.append(
                    sum(1 for x in gaps if x <= summary["small_gap_threshold"]) / len(gaps)
                )
            else:
                per_game_small_ratio.append(0.0)

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        fig.suptitle("NN Top1-Gap Diagnostic", fontsize=14)

        ax = axes[0, 0]
        ax.plot(idx, per_game_gap_mean, marker="o", linewidth=1, label="mean")
        ax.plot(idx, per_game_gap_median, marker="x", linewidth=1, label="median")
        ax.set_title("Top1-Gap by Game")
        ax.set_xlabel("Game Index")
        ax.set_ylabel("Gap")
        ax.legend()

        ax = axes[0, 1]
        if all_gaps:
            bins = min(30, max(8, int(len(all_gaps) ** 0.5)))
            ax.hist(all_gaps, bins=bins, edgecolor="black")
        ax.set_title("Top1-Gap Distribution (State Level)")
        ax.set_xlabel("Top1 - Top2")
        ax.set_ylabel("Count")

        ax = axes[1, 0]
        ax.plot(idx, per_game_small_ratio, marker="o", linewidth=1)
        ax.set_title("Small-Gap Ratio by Game")
        ax.set_xlabel("Game Index")
        ax.set_ylabel("Ratio")

        ax = axes[1, 1]
        ax.axis("off")
        text = (
            f"states={summary['state_count_total']}\n"
            f"small<= {summary['small_gap_threshold']}: "
            f"{summary['small_gap_count']} ({summary['small_gap_ratio']:.2%})\n"
            f"tiny<= {summary['tiny_gap_threshold']}: "
            f"{summary['tiny_gap_count']} ({summary['tiny_gap_ratio']:.2%})\n"
            f"mean={summary['top1_gap_mean']:.4f}\n"
            f"median={summary['top1_gap_median']:.4f}\n"
            f"p10={summary['top1_gap_p10']:.4f}\n"
            f"p90={summary['top1_gap_p90']:.4f}"
        )
        ax.text(0.02, 0.95, text, va="top", ha="left", fontsize=11)

        fig.tight_layout()
        out = Path(output_image_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=140)
        plt.close(fig)
        return out

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
    diag = NNTop1GapDiagnostic(seed=0, depth=2, model_path=None)
    report = diag.run(
        num_games=10,
        base_seed=0,
        small_gap_threshold=0.1,
        tiny_gap_threshold=0.01,
        output_path="artifacts/nn_top1_gap_report_depth2.json",
        output_image_path="artifacts/nn_top1_gap_report_depth2.png",
    )
    s = report["summary"]
    print(
        f"depth={s['depth']} games={s['num_games']} states={s['state_count_total']} "
        f"small={s['small_gap_count']}({s['small_gap_ratio']:.2%}) "
        f"tiny={s['tiny_gap_count']}({s['tiny_gap_ratio']:.2%}) "
        f"p10={s['top1_gap_p10']:.4f} p90={s['top1_gap_p90']:.4f}"
    )


if __name__ == "__main__":
    main()

