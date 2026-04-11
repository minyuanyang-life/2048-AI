import random
from tqdm import tqdm

from src.ai.agent.base_trainer import BaseTrainer
from src.ai.agent.expectimax_agent import ExpectimaxAgent


class ExpectimaxTrainer(BaseTrainer):
    """
    Train heuristic weights, but evaluate them through ExpectimaxAgent.
    """

    def __init__(self, seed: int | None = None) -> None:
        super().__init__(ExpectimaxAgent())
        self._rng = random.Random(seed)

    def train(
            self,
            num_iterations: int = 20,
            num_eval_games: int = 5,
    ) -> None:
        self.agent.load()
        self.pbar = tqdm(range((num_iterations + 1) * num_eval_games), desc="Training")

        best_score = self.evaluate(num_eval_games)
        tqdm.write(f"initial score = {best_score}")

        for index in range(num_iterations):
            self.agent.feedback()
            candidate_score = self.evaluate(num_eval_games)

            if candidate_score > best_score:
                best_score = candidate_score
                self.agent.record_params()
                self.agent.save()
                tqdm.write(f"{index}: improved score = {best_score}")
            else:
                tqdm.write(f"{index}: no improvement")
        self.agent.save()
        tqdm.write(f"final score = {best_score}")


def main() -> None:
    trainer = ExpectimaxTrainer()
    trainer.train()


if __name__ == "__main__":
    main()