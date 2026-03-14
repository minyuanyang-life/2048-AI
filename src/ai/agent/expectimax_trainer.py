import random

from src.ai.agent.base_trainer import BaseTrainer
from src.ai.agent.expectimax_agent import ExpectimaxAgent
from src.ai.evaluator.heuristic_config import (
    HeuristicWeights,
    HeuristicTrainerConfig,
)


class ExpectimaxTrainer(BaseTrainer):
    """
    Train heuristic weights, but evaluate them through ExpectimaxAgent.
    """

    def __init__(self, seed: int | None = None) -> None:
        super().__init__(ExpectimaxAgent())
        self._rng = random.Random(seed)

    def _sample_params(self, params: HeuristicWeights) -> HeuristicWeights:
        params_list = params.to_list()
        sigma = HeuristicTrainerConfig.sample_sigma

        new_params_list = []
        for param in params_list:
            new_param = param + self._rng.gauss(0.0, sigma)
            new_param = min(max(new_param, 0.0), 5.0)
            new_params_list.append(new_param)

        return HeuristicWeights.from_list(new_params_list)

    def _evaluate_params(self, n: int, params: HeuristicWeights) -> float:
        old_params = self.agent.get_params()
        self.agent.set_params(params)
        score = self.evaluate(n)
        self.agent.set_params(old_params)
        return score

    def train(
            self,
            num_iterations: int = 20,
            num_eval_games: int = 5,
    ) -> None:
        self.agent.load()
        best_params = self.agent.get_params()
        best_score = self._evaluate_params(num_eval_games, best_params)
        print(f"initial score = {best_score}")

        for index in range(num_iterations):
            candidate_params = self._sample_params(best_params)
            candidate_score = self._evaluate_params(num_eval_games, candidate_params)

            if candidate_score > best_score:
                best_score = candidate_score
                best_params = candidate_params
                print(f"{index}: improved score = {best_score}")
            else:
                print(f"{index}: no improvement")

        self.agent.set_params(best_params)
        self.agent.save()
        print(f"final score = {best_score}")


def main() -> None:
    trainer = ExpectimaxTrainer()
    trainer.train()


if __name__ == "__main__":
    main()