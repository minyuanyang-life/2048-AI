from time import time
import random

from src.ai.agent.base_trainer import AgentTraining
from src.ai.rl_agent import RLAgent as Agent
from src.ai.config import RLParameters as Parameter
from src.ai.rl_agent_reward import RLAgentReward as Reward
from src.core.enums import GameState
from src.core.game import Game


class RLAgentTraining(AgentTraining):
    def __init__(self, agent: Agent, parameters : Parameter , seed: int = time()) -> None:
        super().__init__(agent, parameters)
        self.reward = Reward()
        self.rng = random.Random(seed)

    def calculate_reward(self, game: Game, new_game: Game) -> float:
        params = [0.3, 0.5, 0.8, 1.5]
        reward = self.reward.calculate_reward(game, new_game, params)
        # print(reward)
        return reward

    def _train_game(self, seed: int) -> None:
        random.seed(seed)
        game = Game()
        while True:
            direction, ranked = self.agent.choose_action(game)
            result, new_game = game.step(direction)
            reward = self.calculate_reward(game, new_game)
            done = (result == GameState.OVER)
            self.agent.train_step(game, new_game, direction, reward, done)
            game = new_game
            if result == GameState.OVER:
                break

    def train(self, N: int, base_seed: int = 0) -> None:
        for i in range(1, N+1):
            self._train_game(base_seed + i)
            self.agent.update_epsilon()
            N_save = 10
            if i % N_save == 0:
                print(f"\t{i}\t/\t{N}\tfinish")

    def run(self, N: int) -> dict:
        file_name = f"qnet.pt"
        self.agent.load_params(file_name)
        print(f"{file_name} loaded")

        self.train(N)

        file_name = f"qnet.pt"
        self.agent.save_params(file_name)
        print(f"{file_name} saved")

def main():
    rl_agent_training = RLAgentTraining(Agent(), Parameter())
    rl_agent_training.run(500)

if __name__ == '__main__':
    main()