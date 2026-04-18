"""
Board module for the 2048 game.

This module defines the Board class, which stores the board in exponent form
and handles tile movement, merging, and spawning.
"""
import random

from src.core.enums import Direction, MoveStatus


class Board:
    """
    Represent the 4x4 board of the 2048 game.

    Responsibilities:
    - Store the board state in exponent form.
    - Handle tile movement, merging, and random tile spawning.
    - Provide read-only access helpers for UI and game logic.

    Notes:
    - 0 means empty.
    - 1 means tile 2, 2 means tile 4, etc.
    """
    def __init__(self) -> None:
        self._grid = [[0 for _ in range(4)] for _ in range(4)]

    def __str__(self) -> str:
        # Define border/separator for stable aligned rendering.
        border = "+------+------+------+------+"
        lines = [border]  # Accumulate output lines.

        # Render each board row.
        for row in self._grid:
            # Empty cells show blank; non-empty cells show centered tile values.
            cells = []
            for num in row:
                if num == 0:
                    # Keep fixed width for alignment.
                    cells.append("      ")
                else:
                    # Display actual tile value with fixed cell width.
                    cells.append(f"{2**num:^6}")

            # Join cells with vertical separators.
            line = "|" + "|".join(cells) + "|"
            lines.append(line)
            lines.append(border)  # Add row separator.

        # Build final multi-line board string.
        return '\n'.join(lines)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Board):
            return NotImplemented
        return self._grid == other._grid

    @property
    def grid(self) -> list[list[int]]:
        return [row.copy() for row in self._grid]

    def clone(self) -> "Board":
        """
        Create and return a deep copy of the current board.

        Returns:
            A new Board instance with the same grid state.
        """
        new_board = Board()
        new_board._grid = [row.copy() for row in self._grid]
        return new_board

    def _get_row(self, row: int) -> list[int]:
        return self._grid[row].copy()

    def _get_column(self, column: int) -> list[int]:
        return [self._grid[j][column] for j in range(4)]

    def get_exponent(self, row: int, col: int) -> int:
        return self._grid[row][col]

    def get_tile_value(self, row: int, col: int) -> int:
        """
        Return the displayed tile value at the given position.

        Args:
            row: Row index of the target cell.
            col: Column index of the target cell.

        Returns:
            The actual tile value shown to the player.
            Returns 0 if the cell is empty.
        """
        exp = self._grid[row][col]
        return 0 if exp == 0 else 2 ** exp

    def set_value(self, row: int, col: int, value: int) -> None:
        """
        Set the tile at the given position using the displayed value.

        Args:
            row: Row index of the target cell.
            col: Column index of the target cell.
            value: Displayed tile value. Must be 0 or a power of 2.

        Raises:
            ValueError: If value is negative or not a power of 2.
        """
        if value == 0:
            self._grid[row][col] = 0
            return

        if value < 0 or (value & (value - 1)) != 0:
            raise ValueError("Tile value must be 0 or a power of 2.")

        self._grid[row][col] = value.bit_length() - 1

    def export_tile_values(self) -> list[list[int]]:
        return [
            [0 if exp == 0 else 2 ** exp for exp in row]
            for row in self._grid
        ]

    def _set_row(self, row: int, new_row: list[int]) -> None:
        self._grid[row] = new_row.copy()

    def _set_column(self, column: int, new_column: list[int]) -> None:
        for j in range(4):
            self._grid[j][column] = new_column[j]

    def is_board_full(self) -> bool:
        return all(cell != 0 for row in self._grid for cell in row)

    def count_empty_tiles(self) -> int:
        return sum(cell == 0 for row in self._grid for cell in row)

    def get_max_exponent(self) -> tuple[int, list[tuple[int, int]]]:
        max_number = -1
        max_index = []
        for i in range(4):
            for j in range(4):
                item = self._grid[i][j]
                if item > max_number:
                    max_number = item
                    max_index = [(i, j)]
                elif item == max_number:
                    max_index.append((i, j))
        return max_number, max_index

    def add_number(self) -> bool:
        """
        Add a random tile to an empty position on the board.

        Returns:
            True if a new tile is added successfully, False if the board is full.
        """
        empty_positions = []
        random_exponent_pool = [2, ] + [1, ] * 9
        for i in range(4):
            for j in range(4):
                if self._grid[i][j] == 0:
                    empty_positions.append((i, j))

        if not empty_positions:
            return False

        random_i, random_j = random.choice(empty_positions)
        random_digit = random.choice(random_exponent_pool)
        self._grid[random_i][random_j] = random_digit
        return True

    def move(self, direction: Direction) -> tuple[MoveStatus, int]:
        """
        Move the board in the given direction.

        Returns:
            A tuple of (move_status, score_gain).
        """
        old_grid = self.clone()
        score = 0
        match direction:
            case Direction.LEFT:
                for i in range(4):
                    line = self._get_row(i)
                    new_line, add_score = self._slide_and_merge(line)
                    self._set_row(i, new_line)
                    score += add_score
            case Direction.RIGHT:
                for i in range(4):
                    line = self._get_row(i)[::-1]
                    new_line, add_score = self._slide_and_merge(line)
                    self._set_row(i, new_line[::-1])
                    score += add_score
            case Direction.UP:
                for i in range(4):
                    line = self._get_column(i)
                    new_line, add_score = self._slide_and_merge(line)
                    self._set_column(i, new_line)
                    score += add_score
            case Direction.DOWN:
                for i in range(4):
                    line = self._get_column(i)[::-1]
                    new_line, add_score = self._slide_and_merge(line)
                    self._set_column(i, new_line[::-1])
                    score += add_score
            case _:
                raise ValueError(f"Invalid direction: {direction}")

        if old_grid == self:
            return MoveStatus.INVALID_MOVE, 0
        return MoveStatus.MOVED, score

    def simulate_move(self, direction: Direction) -> tuple["Board", MoveStatus, int]:
        simulate_board = self.clone()
        move_status, score_delta = simulate_board.move(direction)
        return simulate_board, move_status, score_delta

    def can_move(self) -> bool:
        """
        Return whether the board still has at least one valid move.

        Returns:
            True if the board still has at least one valid move, False otherwise.

        Note: A move is possible if there is at least one empty cell,
        or if two adjacent tiles have the same exponent and can be merged.
        """
        if not self.is_board_full():
            return True

        for i in range(4):
            for j in range(4):
                if i + 1 < 4 and self._grid[i][j] == self._grid[i + 1][j]:
                    return True
                if j + 1 < 4 and self._grid[i][j] == self._grid[i][j + 1]:
                    return True

        return False

    def get_legal_directions(self) -> list[Direction]:
        """
        Return all valid directions that result in a board change.

        Returns:
            A list of directions in which the board can move.
        """
        dirs = []
        for direction in Direction.all():
            board_simulate = self.clone()
            move_status, _ = board_simulate.move(direction)
            if move_status == MoveStatus.MOVED:
                dirs.append(direction)
        return dirs

    @staticmethod
    def _slide_and_merge(line: list[int]) -> tuple[list[int], int]:
        new_line = [0 for _ in range(4)]
        is_merge_ready = True
        j = 0
        score = 0
        for i in range(0, 4):
            if line[i] == 0:
                continue
            if j > 0 and line[i] == new_line[j - 1] and is_merge_ready:
                new_line[j - 1] += 1
                score += 2 ** new_line[j - 1]
                is_merge_ready = False
            else:
                new_line[j] = line[i]
                j += 1
                is_merge_ready = True
        return new_line, score

