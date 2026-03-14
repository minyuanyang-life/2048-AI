from math import log2
from src.core.board import Board
from src.core.game import Game


class RLAgentReward:
    def __init__(self):
        self.reward = 0

    def calculate_reward(self, game: Game, new_game: Game, lamda: list = None) -> float:
        self.reset()
        self.calculate_del_score(game, new_game, lamda = lamda[0])
        self.calculate_del_empty_tiles(game, new_game, lamda = lamda[1])
        self.calculate_del_smooth(game, new_game, lamda = lamda[2])
        self.calculate_del_max_position(game, new_game, lamda = lamda[3])
        return self.reward

    def reset(self):
        self.reward = 0

    @staticmethod
    def _calculate_max_num(board: Board) -> int:
        max_num = 0
        for i in range(4):
            for j in range(4):
                if board._grid[i][j] > max_num:
                    max_num = board._grid[i][j]
        return max_num

    def calculate_del_score(self, game: Game, new_game: Game, lamda: float = 1):
        reward = log2(
            new_game.score
            - game.score
            + 1
        )
        max_num = max(self._calculate_max_num(game.board), self._calculate_max_num(new_game.board))
        reward /= max_num
        reward *= lamda
        self.reward += reward

    def calculate_del_empty_tiles(self, game: Game, new_game: Game, lamda: float = 1):
        reward = (
            new_game.board.count_empty_tiles()
            - game.board.count_empty_tiles()
        )
        reward /= 16
        reward *= lamda
        self.reward += reward

    def _calculate_smooth(self, board: Board) -> float:
        reward = 0
        count = 0
        position = []
        max_num = self._calculate_max_num(board)
        for i in range(4):
            for j in range(4):
                if not i == 3:
                    position.append((i+1, j))
                if not j == 3:
                    position.append((i, j+1))
                for x, y in position:
                    if board._grid[i][j] == 0 or board._grid[x][y] == 0:
                        continue
                    if board._grid[i][j] <= 3 and board._grid[x][y] <= 3:
                        continue
                    count += 1
                    reward += abs(
                        2 ** board._grid[x][y]
                        - 2 ** board._grid[i][j]
                    ) / (
                        2 ** max_num
                    )
        if count == 0:
            return 0
        reward /= count
        return reward


    def calculate_del_smooth(self, game: Game, new_game: Game, lamda: float = 1):
        reward = (
            self._calculate_smooth(new_game.board)
            - self._calculate_smooth(game.board)
        )
        reward *= lamda
        self.reward += reward

    @staticmethod
    def _calculate_max_position(boaed: Board) -> float:
        position = []
        max = 0
        for i in range(4):
            for j in range(4):
                num = boaed._grid[i][j]
                if num > max:
                    max = num
                    position = [(i, j)]
                elif num == max:
                    position.append((i, j))

        reward = 0
        endpoint = (0, 3)
        for x, y in position:
            reward += 0.5 * int( x in endpoint )
            reward += 0.5 * int( y in endpoint )

        reward /= len(position)
        return reward

    def calculate_del_max_position(self, game: Game, new_game: Game, lamda: float = 1):
        reward = (
            self._calculate_max_position(new_game.board)
            - self._calculate_max_position(game.board)
        )
        reward *= lamda
        self.reward += reward