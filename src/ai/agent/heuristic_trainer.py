import random

from src.ai.agent.base_trainer import BaseTrainer
from src.ai.agent.heuristic_agent import HeuristicAgent


class HeuristicTraining(BaseTrainer):
    def __init__(self) -> None:
        super().__init__(HeuristicAgent())
        self._rng = random.Random()

    def train(
            self,
            num_iterations: int = 50,
            num_eval_games: int = 10,
    ) -> None:
        self.agent.load()
        best_score = self.evaluate(num_eval_games)
        print(best_score)

        for _ in range(num_iterations):
            self.agent.feedback()
            score = self.evaluate(num_eval_games)
            if score > best_score:
                best_score = score
                self.agent.record_params()

        self.agent.save()
        print(best_score)

def main():
    heuristic_agent_training = HeuristicTraining()
    heuristic_agent_training.train()

if __name__ == "__main__":
    main()
