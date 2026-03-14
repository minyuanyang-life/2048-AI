"""
Game module for the 2048 game.

This module defines the Game class, which manages the overall game state
and coordinates board updates, score tracking, step counting, and game
status checking.

Responsibilities:
- Initialize a new game with a board and two starting tiles.
- Track the current score and number of valid steps.
- Execute player moves through the board.
- Determine whether the game is won, over, or still running.

Notes:
- The Game class owns a Board instance.
- The Board handles tile movement, merging, and random tile spawning.
- The Game class is responsible for higher-level game flow and status.
"""
from .board import Board
from .enums import Direction, MoveStatus, GameStatus


class Game:
    """
    Represent a single 2048 game session.

    A Game object stores the current board, total score, and number of
    successful steps. It provides methods to check win/lose conditions
    and to execute one move of the game.

    Attributes:
        board: The Board instance storing the tile layout.
        score: The accumulated score of the current game.
        steps: The number of valid moves that have been made.
    """
    def __init__(self):
        self.board = Board()
        self.board.add_number()
        self.board.add_number()
        self.score = 0
        self.steps = 0

    def __str__(self):
        output = str(self.board)
        output += "\nscore: " + str(self.score)
        output += "\nsteps: " + str(self.steps)
        return output

    def is_win(self) -> bool:
        max_exponent, _ = self.board.get_max_exponent()
        return max_exponent >= 15

    def is_over(self) -> bool:
        return not self.board.can_move()

    def step(self, direction: Direction) -> tuple[GameStatus, MoveStatus, dict]:
        """
        Execute one move on the current game.

        Input:
            direction: the move direction.

        Output:
            game_status: result status of the game after this step.
            move_status: result status of this step.
            info: dictionary containing step metadata such as
                  score_delta and done.

        Side effects:
            Mutates self.board, self.score, and self.steps.
        """
        if not isinstance(direction, Direction):
            raise TypeError("direction must be an instance of Direction")

        info_dict = {"score_delta": 0, "done": True}

        if self.is_win():
            return GameStatus.WIN, MoveStatus.INVALID_MOVE, info_dict
        if self.is_over():
            return GameStatus.GAME_OVER, MoveStatus.INVALID_MOVE, info_dict


        move_status, score_delta = self.board.move(direction)
        info_dict["score_delta"] = score_delta
        if move_status == MoveStatus.INVALID_MOVE:
            info_dict["done"] = False
            return GameStatus.RUNNING, move_status, info_dict

        self.score += score_delta
        self.steps += 1

        if self.is_win():
            return GameStatus.WIN, move_status, info_dict

        if not self.board.add_number():
            raise RuntimeError(
                "Invariant violated: add_number() failed after a valid move."
            )

        if self.is_over():
            return GameStatus.GAME_OVER, move_status, info_dict

        info_dict["done"] = False
        return GameStatus.RUNNING, move_status, info_dict

    def simulate_step(self, direction: Direction) -> tuple["Game", GameStatus, MoveStatus, dict]:
        """
        Simulate to execute one move on the current game.

        Input:
            direction: the move direction.

        Output:
            game_status: result status of the game after this step.
            move_status: result status of this step.
            info: dictionary containing step metadata such as
                  score_delta and done.
        """
        simulate_game = self.clone()
        game_status, move_status, info_dict = simulate_game.step(direction)
        return simulate_game, game_status, move_status, info_dict

    def clone(self) -> "Game":
        """
        Create a deep copy of the current game state.

        Output:
            A new Game object with the same board, score, and step count.

        Notes:
            Mutating the cloned game does not affect the original game.
        """
        new_game = Game.__new__(Game)
        new_game.board = self.board.clone()
        new_game.score = self.score
        new_game.steps = self.steps
        return new_game

