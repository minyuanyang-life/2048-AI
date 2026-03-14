import os
import random
from copy import deepcopy
from operator import itemgetter
from time import time
import torch
from torch import Tensor, optim, nn

from src.ai.agent import Agent
from src.ai.config import RLParameters
from src.ai.q_network import QNetwork
from src.core.enums import Direction, GameState
from src.core.game import Game
from src.core.move import MoveEngine


class RLAgent(Agent):
    def __init__(self, seed: int = time()) -> None:
        self.params = RLParameters()
        self.rng = random.Random(seed)

        self.q_network = QNetwork(input_size=16, output_size=4)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)

    def save_params(self, file_name: str = "qnet.pt") -> None:
        if not os.path.exists("/src/ai\\data"):
            os.mkdir("/src/ai\\data")
        file_path = "C:\\Work\\github\\2048\\src\\ai\\data\\" + file_name
        torch.save(self.q_network.state_dict(), file_path)

    def load_params(self, file_name: str = "qnet.pt") -> None:
        file_path = "C:\\Work\\github\\2048\\src\\ai\\data\\" + file_name
        self.q_network.load_state_dict(torch.load(file_path, map_location="cpu"))

    @staticmethod
    def get_illegal_direction(game: Game) -> list[Direction]:
        dirs = []
        for direction in Direction.all():
            result, _, _ = MoveEngine.apply(game.board, direction)
            if result == GameState.SUCCESS:
                dirs.append(direction)
        return dirs

    def choose_action(self, game: Game, evaluate: bool = False) -> tuple[Direction, list | None]:
        dirs = self.get_illegal_direction(game)
        for direction in Direction.all():
            result, _, _ = MoveEngine.apply(game.board, direction)
            if result == GameState.SUCCESS:
                dirs.append(direction)
        if self.rng.uniform(0, 1) < self.params.epsilon and not evaluate:
            return self.rng.choice(dirs), None
        else:
            return self._get_best_action(game)

    def _get_best_action(self, game: Game) -> tuple[Direction, list]:
        game_simulate = deepcopy(game)
        state = self.get_state(game_simulate)
        q_value = self.q_network(state).detach().cpu().tolist()
        scores = dict(zip(Direction.all(), list(q_value)))
        min_q = min(list(q_value))
        dirs = self.get_illegal_direction(game_simulate)
        for action in scores.keys():
            if action not in dirs:
                scores[action] = min_q - 1000
        target_direction = max(scores, key=scores.get)
        ranked = sorted(scores.items(), key=itemgetter(1), reverse=True)
        return target_direction, ranked

    def update_epsilon(self):
        self.params.epsilon = max(0.01, self.params.epsilon * self.params.epsilon_decay)

    def train_step(self,
                   game: Game,
                   next_game: Game,
                   action: Direction,
                   reward: float,
                   done: bool
                   ) -> None:
        state = self.get_state(game)
        next_state = self.get_state(next_game)

        q_pred = self.q_network(state)[Direction.dir_to_num(action)]

        with torch.no_grad():
            q_next = self.q_network(next_state).max()
            q_target = (
                q_pred.new_tensor(reward) +
                self.params.gamma * q_next * ( 1 - done )
            )

        loss = nn.MSELoss()(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    @staticmethod
    def get_state(game: Game) -> Tensor:
        num_list = []
        for row in game.board._grid:
            for cell in row:
                num_list.append(cell)
        return torch.tensor(num_list, dtype=torch.float32)

