import random

from src.ai.agent.base_trainer import BaseTrainer
from src.ai.agent.heuristic_agent import HeuristicAgent
from src.ai.evaluator.heuristic_config import HeuristicWeights, HeuristicTrainerConfig


class HeuristicTraining(BaseTrainer):
    def __init__(self) -> None:
        super().__init__(HeuristicAgent())
        self._rng = random.Random()

    def _sample_params(self, params: HeuristicWeights) -> HeuristicWeights:
        params_list = params.to_list()
        sample_sigma = HeuristicTrainerConfig.sample_sigma
        new_params_list = []
        for param in params_list:
            new_param = param + self._rng.gauss(0.0, sample_sigma)
            new_param = min(max(new_param, 0.0), 5.0)
            new_params_list.append(new_param)
        return HeuristicWeights.from_list(new_params_list)

    def _evaluate_params(
            self,
            n: int,
            params: HeuristicWeights
    ) -> float:
        old_params = self.agent.get_params()
        self.agent.set_params(params)
        score = self.evaluate(n)
        self.agent.set_params(old_params)
        return score


    def train(
            self,
            num_iterations: int = 50,
            num_eval_games: int = 10,
    ) -> None:
        self.agent.load()

        candidate_params = HeuristicWeights()
        best_score = self._evaluate_params(num_iterations, candidate_params)
        best_params = candidate_params
        print(best_score)

        for _ in range(num_eval_games):
            candidate_params = self._sample_params(candidate_params)
            score = self._evaluate_params(num_iterations, candidate_params)
            if score > best_score:
                best_score = score
                best_params = candidate_params

        self.agent.save(best_params)
        print(best_score)

def main():
    heuristic_agent_training = HeuristicTraining()
    heuristic_agent_training.train()

if __name__ == "__main__":
    main()
