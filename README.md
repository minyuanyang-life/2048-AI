# 2048-AI

This is a 2048 UI + AI project written in Python.

The project started from implementing the 2048 game itself, including the core game logic and the graphical interface. After completing these basic parts, I further experimented with adding AI modules so that the program could play the game automatically. In this process, I roughly divided the system into several parts: agent, evaluator, and trainer / tester modules for training and testing.

## What is included

This project currently includes:

- the core logic of the 2048 game
- a Tkinter-based graphical interface
- several agents, including a random agent and an expectimax agent
- several board evaluators, including a heuristic evaluator and a fully connected neural network evaluator
- training, testing, and diagnostic scripts

## What I mainly worked on

The main parts I worked on include:

- independently implementing the core logic of the 2048 game
- building the graphical interface
- designing and debugging different evaluators
- experimenting with training and evaluation pipelines
- observing some practical issues during training, such as unstable improvement or performance plateaus

In this project, the core logic of 2048 was mainly implemented independently by me. In the development of some agent modules, I also used AI assistance, mainly for implementation, debugging, and organizing parts of the code.

## Notes on the codebase

At present, the core part is already relatively well-structured and encapsulated, with clearer organization and more consistent comments. In comparison, the comments in some other parts are still relatively sparse, and the level of code organization is not yet fully consistent. These parts still need further improvement.

## Project Structure

```text
src/
  core/                 # core logic of 2048
  ui/                   # Tkinter GUI
  ai/
    agent/              # agents, trainers
    evaluator/          # heuristic and neural evaluators
    optimizer/          # optimization utilities
    metrics/            # profiling / metrics
    diagnostics/        # diagnostic scripts
tests/                  # tests
docs/                   # design notes
run.py                  # GUI entry point
```

## How to run

### Environment
- Python 3.10+
- Dependencies:
pip install torch tqdm matplotlib

### Run the GUI
python run.py

### Train the neural evaluator
python -m src.ai.agent.NN_trainer

### Run tests
python -m src.ai.agent.NN_tester
python -m src.ai.agent.heuristic_tester

### Run diagnostic scripts
python -m src.ai.agent.teacher_diagnostics
python -m src.ai.diagnostics.nn_top1_gap_diagnostic

## Current status

This project is still being improved. At the moment, it already has a relatively complete code structure and basic training and testing pipelines, but there is still a lot of room for improvement in model performance, training stability, evaluation design, and code organization.

## Remarks

- Runtime outputs are stored under `artifacts/`
- The overall code structure can still be understood even without model checkpoint files
- Some trained weights and result reports may be added later
