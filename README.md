# 2048-AI

A Python implementation of 2048 with:
- complete game core logic,
- Tkinter GUI,
- multiple AI agents (random / heuristic / expectimax + NN evaluator),
- training, testing, and diagnostics scripts.

## Features

- Core 2048 engine (`Board`, `Game`, move/merge/spawn rules)
- GUI play mode (`run.py`)
- AI agents:
  - `RandomAgent`
  - `HeuristicAgent`
  - `ExpectimaxAgent` + `HeuristicEvaluator`
  - `ExpectimaxAgent` + `NNEvaluator`
- Training modules:
  - heuristic/expectimax parameter training
  - NN trainer with teacher-guided pretrain options
- Testing / diagnostics:
  - NN tester and heuristic tester
  - teacher alignment diagnostics
  - top1-gap diagnostic (decision confidence)

## Project Structure

```text
src/
  core/                 # 2048 game core (board, game, enums)
  ui/                   # GUI (Tkinter)
  ai/
    agent/              # agents, trainers, testers
    evaluator/          # heuristic and neural evaluators
    optimizer/          # optimization helpers (e.g. SPSA)
    metrics/            # profiling/metrics data structures
    diagnostics/        # analysis-only diagnostic scripts
tests/                  # tests
docs/                   # design notes
run.py                  # GUI entry point
```

## Quick Start

### 1) Environment

- Python 3.10+ recommended
- Install dependencies (if needed in your local environment):

```bash
pip install torch tqdm matplotlib
```

### 2) Run GUI

```bash
python run.py
```

### 3) Train NN

```bash
python -m src.ai.agent.NN_trainer
```

### 4) Test NN / Heuristic

```bash
python -m src.ai.agent.NN_tester
python -m src.ai.agent.heuristic_tester
```

### 5) Diagnostics

```bash
python -m src.ai.agent.teacher_diagnostics
python -m src.ai.diagnostics.nn_top1_gap_diagnostic
```

## Artifacts & Checkpoints

- Runtime artifacts and reports are written under `artifacts/`.
- Model/checkpoint files are intentionally not required for code structure publication.
- If you want to test a specific model, configure `model_path` / `load_model_path` in tester/trainer entry scripts.

## Notes

- The repository currently focuses on code structure and reproducible pipelines.
- Trained weights can be published later as release assets if needed.
