import random
from tqdm import tqdm

from src.ai.agent.base_trainer import BaseTrainer
from src.ai.agent.expectimax_agent import ExpectimaxAgent
from src.ai.optimizer.spsa_optimizer import SPSAOptimizer


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
            early_stop_patience: int | None = None,
            min_improve: float = 0.0,
    ) -> None:
        self.agent.load()
        self.pbar = tqdm(range((1 + 2 * num_iterations) * num_eval_games), desc="Training")
        optimizer = SPSAOptimizer(
            self.agent.get_param_vector(),
            early_stop_patience=early_stop_patience,
            min_improve=min_improve,
        )

        best_score = self.evaluate(num_eval_games)
        optimizer.observe(best_score)
        tqdm.write(f"initial score = {best_score}")

        for index in range(num_iterations):
            proposal = optimizer.propose()

            self.agent.set_param_vector(proposal.plus)
            score_plus = self.evaluate(num_eval_games)

            self.agent.set_param_vector(proposal.minus)
            score_minus = self.evaluate(num_eval_games)

            if score_plus >= score_minus:
                candidate_score = score_plus
                candidate_params = proposal.plus
            else:
                candidate_score = score_minus
                candidate_params = proposal.minus

            self.agent.set_param_vector(optimizer.update(proposal, score_plus, score_minus))

            if candidate_score > best_score + min_improve:
                best_score = candidate_score
                self.agent.set_param_vector(candidate_params)
                self.agent.record_params()
                self.agent.save()
                self.agent.set_param_vector(optimizer.get_params())
                tqdm.write(f"{index}: improved score = {best_score}")
            else:
                tqdm.write(f"{index}: no improvement")
            optimizer.observe(candidate_score)
            if optimizer.should_stop():
                tqdm.write(
                    f"early stop at iteration {index}, "
                    f"no improvement for {optimizer.no_improve_rounds()} rounds"
                )
                break
        self.agent.save()
        tqdm.write(f"final score = {best_score}")


def main() -> None:
    trainer = ExpectimaxTrainer()
    trainer.train(
        num_iterations=200,
        num_eval_games=5,
        early_stop_patience=10,
        min_improve=50.0,
    )


if __name__ == "__main__":
    main()
