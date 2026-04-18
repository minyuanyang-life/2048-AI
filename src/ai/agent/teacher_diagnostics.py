import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm

from src.ai.evaluator.NN_evaluator import NNEvaluator
from src.ai.evaluator.heuristic_evaluator import HeuristicEvaluator
from src.core.board import Board
from src.core.enums import Direction, GameStatus, MoveStatus
from src.core.game import Game


@dataclass
class StateSample:
    board: Board
    features: list[float]
    teacher_target: float
    max_exponent: int


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _board_features(board: Board) -> list[float]:
    return [float(board.get_exponent(r, c)) for r in range(4) for c in range(4)]


def _collect_teacher_dataset(
    teacher: HeuristicEvaluator,
    num_games: int,
    max_steps_per_game: int,
    base_seed: int,
    target_scale: float,
    rollout_policy: str = "teacher_greedy",
) -> list[StateSample]:
    samples: list[StateSample] = []
    rng = random.Random(base_seed + 2024)
    for game_idx in tqdm(range(num_games), desc="Collect Diagnostic Data"):
        _set_global_seed(base_seed + game_idx)
        game = Game()
        game_status = GameStatus.RUNNING
        steps = 0
        while game_status == GameStatus.RUNNING and steps < max_steps_per_game:
            board = game.board.clone()
            max_exp, _ = board.get_max_exponent()
            samples.append(
                StateSample(
                    board=board,
                    features=_board_features(board),
                    teacher_target=float(teacher.evaluate_board(board)) / target_scale,
                    max_exponent=int(max_exp),
                )
            )
            legal_dirs = game.board.get_legal_directions()
            if not legal_dirs:
                break
            if rollout_policy == "random":
                direction = rng.choice(legal_dirs)
            elif rollout_policy == "teacher_greedy":
                best_dir = None
                best_score = float("-inf")
                for d in legal_dirs:
                    next_board, move_status, _ = game.board.simulate_move(d)
                    if move_status != MoveStatus.MOVED:
                        continue
                    s = float(teacher.evaluate_board(next_board))
                    if s > best_score:
                        best_score = s
                        best_dir = d
                if best_dir is None:
                    break
                direction = best_dir
            else:
                raise ValueError(f"Unsupported rollout_policy: {rollout_policy}")
            game_status, move_status, _ = game.step(direction)
            if move_status == MoveStatus.INVALID_MOVE:
                raise RuntimeError("Random policy produced invalid move unexpectedly.")
            steps += 1
    return samples


def _to_tensors(samples: list[StateSample], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor([s.features for s in samples], dtype=torch.float32, device=device)
    y = torch.tensor([s.teacher_target for s in samples], dtype=torch.float32, device=device).unsqueeze(1)
    return x, y


def _evaluate_mse(model: NNEvaluator, x: torch.Tensor, y: torch.Tensor, batch_size: int) -> float:
    if x.shape[0] == 0:
        return 0.0
    model.set_eval_mode()
    losses = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = x[start:start + batch_size]
            yb = y[start:start + batch_size]
            pred = model.forward(xb)
            loss = model.criterion(pred, yb)
            losses.append(float(loss.item()))
    return float(sum(losses) / len(losses))


def _train_one_epoch(model: NNEvaluator, x: torch.Tensor, y: torch.Tensor, batch_size: int) -> float:
    model.set_train_mode()
    n = x.shape[0]
    perm = torch.randperm(n, device=model.device)
    losses = []
    for start in range(0, n, batch_size):
        idx = perm[start:start + batch_size]
        xb = x[idx]
        yb = y[idx]
        pred = model.forward(xb)
        loss = model.criterion(pred, yb)
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()
        losses.append(float(loss.item()))
    return float(sum(losses) / len(losses))


def _average_ranks(values: list[float]) -> list[float]:
    indexed = list(enumerate(values))
    indexed.sort(key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def _pearson(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den_x = math.sqrt(sum((a - mx) ** 2 for a in x))
    den_y = math.sqrt(sum((b - my) ** 2 for b in y))
    if den_x == 0.0 or den_y == 0.0:
        return 0.0
    return num / (den_x * den_y)


def _spearman(x: list[float], y: list[float]) -> float:
    return _pearson(_average_ranks(x), _average_ranks(y))


def _direction_ranking(scores: dict[Direction, float]) -> list[Direction]:
    return sorted(scores.keys(), key=lambda d: (-scores[d], int(d)))


def _pairwise_agreement(teacher_scores: dict[Direction, float], nn_scores: dict[Direction, float]) -> float:
    dirs = list(teacher_scores.keys())
    agree = 0
    total = 0
    eps = 1e-9
    for i in range(len(dirs)):
        for j in range(i + 1, len(dirs)):
            di = dirs[i]
            dj = dirs[j]
            td = teacher_scores[di] - teacher_scores[dj]
            nd = nn_scores[di] - nn_scores[dj]
            if abs(td) <= eps or abs(nd) <= eps:
                continue
            total += 1
            if (td > 0 and nd > 0) or (td < 0 and nd < 0):
                agree += 1
    if total == 0:
        return 1.0
    return agree / total


def _stage_name(max_exp: int, early_max: int = 7, mid_max: int = 10) -> str:
    if max_exp <= early_max:
        return "early"
    if max_exp <= mid_max:
        return "mid"
    return "late"


def _stage_counts(
    samples: list[StateSample],
    early_max: int = 7,
    mid_max: int = 10,
) -> dict[str, int]:
    counts = {"early": 0, "mid": 0, "late": 0}
    for s in samples:
        counts[_stage_name(s.max_exponent, early_max, mid_max)] += 1
    return counts


def _stratified_split(
    samples: list[StateSample],
    train_ratio: float,
    rng: random.Random,
    early_max: int = 7,
    mid_max: int = 10,
) -> tuple[list[StateSample], list[StateSample]]:
    by_stage = {"early": [], "mid": [], "late": []}
    for s in samples:
        by_stage[_stage_name(s.max_exponent, early_max, mid_max)].append(s)

    train_samples: list[StateSample] = []
    val_samples: list[StateSample] = []
    for stage in ("early", "mid", "late"):
        group = by_stage[stage]
        if not group:
            continue
        rng.shuffle(group)
        split = int(len(group) * train_ratio)
        # keep at least 1 sample in val when possible
        if len(group) >= 2:
            split = min(max(split, 1), len(group) - 1)
        else:
            split = 1
        train_samples.extend(group[:split])
        val_samples.extend(group[split:])

    if not train_samples and val_samples:
        train_samples = val_samples[:-1] or val_samples
    if not val_samples and train_samples:
        val_samples = train_samples[-1:]
        train_samples = train_samples[:-1] or train_samples
    return train_samples, val_samples


def _score_boards(model: NNEvaluator, boards: list[Board]) -> list[float]:
    return model.evaluate_boards(boards)


def _alignment_metrics(
    model: NNEvaluator,
    teacher: HeuristicEvaluator,
    samples: list[StateSample],
    target_scale: float,
    early_max: int = 7,
    mid_max: int = 10,
) -> dict:
    if not samples:
        return {}

    nn_board_scores = _score_boards(model, [s.board for s in samples])
    teacher_board_scores = [s.teacher_target for s in samples]
    pearson_all = _pearson(nn_board_scores, teacher_board_scores)
    spearman_all = _spearman(nn_board_scores, teacher_board_scores)

    action_exact = []
    action_top1 = []
    action_pairwise = []
    by_stage = {"early": [], "mid": [], "late": []}

    for sample in tqdm(samples, desc="Action Ranking Check"):
        legal_dirs = sample.board.get_legal_directions()
        if len(legal_dirs) < 2:
            continue
        successors = []
        for d in legal_dirs:
            next_board, move_status, _ = sample.board.simulate_move(d)
            if move_status == MoveStatus.MOVED:
                successors.append((d, next_board))
        if len(successors) < 2:
            continue

        succ_boards = [b for _, b in successors]
        nn_succ = _score_boards(model, succ_boards)
        teacher_succ = [float(teacher.evaluate_board(b)) / target_scale for b in succ_boards]

        nn_scores = {d: s for (d, _), s in zip(successors, nn_succ)}
        teacher_scores = {d: s for (d, _), s in zip(successors, teacher_succ)}

        nn_rank = _direction_ranking(nn_scores)
        teacher_rank = _direction_ranking(teacher_scores)

        exact = 1.0 if nn_rank == teacher_rank else 0.0
        top1 = 1.0 if nn_rank[0] == teacher_rank[0] else 0.0
        pairwise = _pairwise_agreement(teacher_scores, nn_scores)

        action_exact.append(exact)
        action_top1.append(top1)
        action_pairwise.append(pairwise)
        by_stage[_stage_name(sample.max_exponent, early_max, mid_max)].append((exact, top1, pairwise))

    def _mean(v: list[float]) -> float:
        return float(sum(v) / len(v)) if v else 0.0

    stage_metrics = {}
    for stage, rows in by_stage.items():
        if not rows:
            stage_metrics[stage] = {
                "count": 0,
                "exact_match_rate": 0.0,
                "top1_match_rate": 0.0,
                "pairwise_agreement": 0.0,
                "pearson": 0.0,
                "spearman": 0.0,
            }
            continue
        idx = [
            i for i, s in enumerate(samples)
            if _stage_name(s.max_exponent, early_max, mid_max) == stage
        ]
        nn_stage = [nn_board_scores[i] for i in idx]
        teacher_stage = [teacher_board_scores[i] for i in idx]
        stage_metrics[stage] = {
            "count": len(rows),
            "exact_match_rate": _mean([r[0] for r in rows]),
            "top1_match_rate": _mean([r[1] for r in rows]),
            "pairwise_agreement": _mean([r[2] for r in rows]),
            "pearson": _pearson(nn_stage, teacher_stage),
            "spearman": _spearman(nn_stage, teacher_stage),
        }

    return {
        "board_score_correlation": {
            "pearson": pearson_all,
            "spearman": spearman_all,
        },
        "action_ranking": {
            "samples_with_2plus_moves": len(action_exact),
            "exact_match_rate": _mean(action_exact),
            "top1_match_rate": _mean(action_top1),
            "pairwise_agreement": _mean(action_pairwise),
        },
        "stage_metrics": stage_metrics,
    }


def run_teacher_diagnostics(
    seed: int = 0,
    num_games: int = 80,
    max_steps_per_game: int = 200,
    epochs: int = 8,
    batch_size: int = 256,
    train_ratio: float = 0.8,
    target_scale: float = 10.0,
    base_seed: int = 300_000,
    rollout_policy: str = "teacher_greedy",
    early_stage_max: int = 7,
    mid_stage_max: int = 10,
    auto_relax_late_stage: bool = True,
    late_min_samples: int = 200,
    late_collect_games_per_round: int = 30,
    late_collect_max_rounds: int = 10,
    output_path: str | Path = "artifacts/teacher_diagnostic_report.json",
) -> dict:
    _set_global_seed(seed)

    model = NNEvaluator()
    try:
        model.load()
    except FileNotFoundError:
        pass

    teacher = HeuristicEvaluator()
    try:
        teacher.load()
    except FileNotFoundError:
        pass

    samples = _collect_teacher_dataset(
        teacher=teacher,
        num_games=num_games,
        max_steps_per_game=max_steps_per_game,
        base_seed=base_seed,
        target_scale=target_scale,
        rollout_policy=rollout_policy,
    )
    if not samples:
        raise RuntimeError("No samples collected.")

    used_mid_stage_max = int(mid_stage_max)

    # Ensure enough late-stage coverage for reliable diagnostics.
    stage_counts = _stage_counts(samples, early_stage_max, used_mid_stage_max)
    if stage_counts["late"] < late_min_samples:
        needed = late_min_samples - stage_counts["late"]
        tqdm.write(
            f"late samples too few ({stage_counts['late']}); "
            f"try collecting ~{needed} more late samples"
        )
        for extra_round in range(late_collect_max_rounds):
            extra_samples = _collect_teacher_dataset(
                teacher=teacher,
                num_games=late_collect_games_per_round,
                max_steps_per_game=max_steps_per_game,
                base_seed=base_seed + 10_000_000 + extra_round * 100_000,
                target_scale=target_scale,
                rollout_policy="teacher_greedy",
            )
            late_only = [
                s for s in extra_samples
                if _stage_name(s.max_exponent, early_stage_max, mid_stage_max) == "late"
            ]
            if late_only:
                samples.extend(late_only)
            stage_counts = _stage_counts(samples, early_stage_max, used_mid_stage_max)
            if stage_counts["late"] >= late_min_samples:
                break

    # Fallback: relax stage boundary for diagnostics when late is still empty.
    if auto_relax_late_stage and stage_counts["late"] == 0:
        while used_mid_stage_max > early_stage_max:
            used_mid_stage_max -= 1
            relaxed_counts = _stage_counts(samples, early_stage_max, used_mid_stage_max)
            if relaxed_counts["late"] > 0:
                stage_counts = relaxed_counts
                tqdm.write(
                    "late still empty after extra collection; "
                    f"relax diagnostic stage boundary to mid<= {used_mid_stage_max}"
                )
                break

    rng = random.Random(seed + 99)
    train_samples, val_samples = _stratified_split(
        samples, train_ratio, rng, early_stage_max, used_mid_stage_max
    )
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)

    x_train, y_train = _to_tensors(train_samples, model.device)
    x_val, y_val = _to_tensors(val_samples, model.device)

    history = []
    train0 = _evaluate_mse(model, x_train, y_train, batch_size)
    val0 = _evaluate_mse(model, x_val, y_val, batch_size)
    history.append({"epoch": 0, "train_loss": train0, "val_loss": val0})

    for ep in tqdm(range(1, epochs + 1), desc="Teacher Diagnostic Train"):
        _train_one_epoch(model, x_train, y_train, batch_size)
        train_l = _evaluate_mse(model, x_train, y_train, batch_size)
        val_l = _evaluate_mse(model, x_val, y_val, batch_size)
        history.append({"epoch": ep, "train_loss": train_l, "val_loss": val_l})

    train_last = history[-1]["train_loss"]
    val_last = history[-1]["val_loss"]
    min_val = min(h["val_loss"] for h in history)
    min_val_epoch = min(history, key=lambda h: h["val_loss"])["epoch"]
    overfit_flag = (val_last > min_val * 1.05) and (train_last < history[min_val_epoch]["train_loss"])

    # Reload baseline for "before" alignment.
    try:
        align_model_before = NNEvaluator()
        align_model_before.load()
    except FileNotFoundError:
        align_model_before = NNEvaluator()
    align_before = _alignment_metrics(
        model=align_model_before,
        teacher=teacher,
        samples=val_samples,
        target_scale=target_scale,
        early_max=early_stage_max,
        mid_max=used_mid_stage_max,
    )
    align_after = _alignment_metrics(
        model=model,
        teacher=teacher,
        samples=val_samples,
        target_scale=target_scale,
        early_max=early_stage_max,
        mid_max=used_mid_stage_max,
    )

    report = {
        "config": {
            "seed": seed,
            "num_games": num_games,
            "max_steps_per_game": max_steps_per_game,
            "epochs": epochs,
            "batch_size": batch_size,
            "train_ratio": train_ratio,
            "target_scale": target_scale,
            "base_seed": base_seed,
            "rollout_policy": rollout_policy,
            "early_stage_max": early_stage_max,
            "mid_stage_max": mid_stage_max,
            "used_mid_stage_max": used_mid_stage_max,
            "auto_relax_late_stage": auto_relax_late_stage,
            "late_min_samples": late_min_samples,
            "late_collect_games_per_round": late_collect_games_per_round,
            "late_collect_max_rounds": late_collect_max_rounds,
        },
        "dataset": {
            "total_samples": len(samples),
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "stage_counts_total": _stage_counts(samples, early_stage_max, used_mid_stage_max),
            "stage_counts_train": _stage_counts(train_samples, early_stage_max, used_mid_stage_max),
            "stage_counts_val": _stage_counts(val_samples, early_stage_max, used_mid_stage_max),
        },
        "loss": {
            "history": history,
            "train_drop": train0 - train_last,
            "val_drop": val0 - val_last,
            "min_val_loss": min_val,
            "min_val_epoch": min_val_epoch,
            "overfit_flag": bool(overfit_flag),
        },
        "alignment_before": align_before,
        "alignment_after": align_after,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def main() -> None:
    report = run_teacher_diagnostics(
        seed=0,
        num_games=400,
        max_steps_per_game=10000,
        epochs=5,
        batch_size=256,
        train_ratio=0.8,
        target_scale=10.0,
        base_seed=300_000,
        output_path="artifacts/teacher_diagnostic_report.json",
    )
    loss = report["loss"]
    after = report["alignment_after"]["action_ranking"]
    print(
        f"samples={report['dataset']['total_samples']} "
        f"train_drop={loss['train_drop']:.6f} "
        f"val_drop={loss['val_drop']:.6f} "
        f"overfit={loss['overfit_flag']} "
        f"top1={after['top1_match_rate']:.3f} "
        f"exact={after['exact_match_rate']:.3f}"
    )


if __name__ == "__main__":
    main()
