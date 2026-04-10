import tkinter as tk
from src.ai.agent.agent import Agent, TrainableAgent
from src.core.enums import GameStatus, Direction
from src.core.game import Game
from src.ui.config import LayoutConfig

class GameGUI:
    def __init__(self, root, title, agent: Agent):
        self.root = root
        self.root.title(title)
        self.root.resizable(False, False)
        self.game = Game()
        self.layout = LayoutConfig()
        self.agent = agent
        self.auto_mode = False

        self._setup_agent()
        self._create_canvas()
        self._bind_keys()

    def _setup_agent(self):
        pass
        # if isinstance(self.agent, RLAgent):
        #     self.agent.load_params("qnet.pt")

    def _create_canvas(self):
        self.canvas = tk.Canvas(
            self.root,
            width=self.layout.canvas_width,
            height=self.layout.canvas_height,
            bg="#bbada0"
        )
        self.canvas.pack(padx=0, pady=0)

    def _bind_keys(self):
        self.root.bind("<a>", lambda e: self.toggle_auto())
        self.root.bind("<Left>", lambda e: self.step(Direction.LEFT))
        self.root.bind("<Right>", lambda e: self.step(Direction.RIGHT))
        self.root.bind("<Up>", lambda e: self.step(Direction.UP))
        self.root.bind("<Down>", lambda e: self.step(Direction.DOWN))
        self.root.bind("<Escape>", lambda e: self.root.quit())

    def get_color(self, num):
        if num > 13:
            num = 13
        return self.layout.color_map[num]

    def _draw_grid(self):
        for i in range(4):
            for j in range(4):
                self._draw_cell(i, j)

    def _draw_cell(self, i, j):
        x0 = self.layout.padding + (self.layout.gap + self.layout.cell_size) * j
        y0 = self.layout.padding + (self.layout.gap + self.layout.cell_size) * i
        x1 = x0 + self.layout.cell_size
        y1 = y0 + self.layout.cell_size
        num = self.game.board.grid[i][j]
        self.canvas.create_rectangle(
            x0, y0, x1, y1,
            fill=self.get_color(num)[0],
            outline="",
            width=0
        )
        if num != 0:
            x_text = x0 + self.layout.cell_size / 2
            y_text = y0 + self.layout.cell_size / 2
            self._draw_text(x_text, y_text, num)

    def _draw_text(self, x, y, num):
        self.canvas.create_text(
            x, y,
            text=str(2 ** num),
            font=("Arial", 24, "bold"),
            fill=self.get_color(num)[1]
        )

    def _get_agent_suggestion(self) -> Direction | None:
        if self.agent is None:
            return None
        return self.agent.get_action(self.game)

    def _get_agent_ranked(self) -> list[tuple[Direction, float]] | None:
        if self.agent is None:
            return None
        if not self.agent.supports_action_ranking():
            return None
        return self.agent.get_action_ranking(self.game)

    def _draw_info(self, show_suggestion: bool = True) -> None:
        score_x = self.layout.canvas_width - self.layout.sidebar_width / 2
        score_y = self.layout.canvas_height * 1 / 6
        self.canvas.create_text(
            score_x, score_y,
            text=f"score:{self.game.score}",
            font=("Arial", 18, "bold"),
            fill="#904020"
        )

        steps_x = self.layout.canvas_width - self.layout.sidebar_width / 2
        steps_y = self.layout.canvas_height * 2 / 6
        self.canvas.create_text(
            steps_x, steps_y,
            text=f"steps:{self.game.steps}",
            font=("Arial", 18, "bold"),
            fill="#777777"
        )
        if show_suggestion:
            direction = self._get_agent_suggestion()
            ranked = self._get_agent_ranked()
        else:
            direction = None
            ranked = None

        ranked_x = self.layout.canvas_width - self.layout.sidebar_width / 2
        ranked_y = self.layout.canvas_height * 4.5 / 6
        ranked_string = ""
        if not ranked is None:
            for item in ranked:
                if item[1] is None:
                    ranked_string += f"{item[0]}\tNone\n"
                ranked_string += f"{item[0]}\t{item[1]:.2f}\n"

        self.canvas.create_text(
            ranked_x, ranked_y,
            text=ranked_string,
            font=("Arial", 12, "bold"),
            fill="#777777"
        )

        direction_x = self.layout.canvas_width - self.layout.sidebar_width / 2
        direction_y = self.layout.canvas_height * 3 / 6
        self.canvas.create_text(
            direction_x, direction_y,
            text=f"suggestion:{direction}",
            font=("Arial", 18, "bold"),
            fill="#F0E68C"
        )

    def render(self, show_suggestion: bool = True) -> None:
        self.canvas.delete("all")
        self._draw_grid()
        self._draw_info(show_suggestion)

    def step(self, direction):
        game_status, _, _ = self.game.step(direction)

        if game_status == GameStatus.GAME_OVER:
            self.auto_mode = False
            self.render(show_suggestion=False)
            self._show_overlay("Over")
            return

        if game_status == GameStatus.WIN:
            self.auto_mode = False
            self.render(show_suggestion=False)
            self._show_overlay("Win")
            return

        self.render()

    def toggle_auto(self):
        self.auto_mode = not self.auto_mode
        if self.auto_mode:
            self._auto_loop()

    def _auto_loop(self):
        if not self.auto_mode:
            return

        direction = self._get_agent_suggestion()
        self.step(direction)

        self.root.after(20, self._auto_loop)

    def run(self):
        if isinstance(self.agent, TrainableAgent):
            self.agent.load()

        self.render(show_suggestion=False)
        self.root.mainloop()

    def _show_overlay(self, text, color="#ff0000"):
        self.canvas.create_text(
            self.layout.canvas_width / 2,
            self.layout.canvas_height / 2,
            text=text,
            font=("Arial", 30, "bold"),
            fill=color
        )

def run(agent_name: str = "random") -> None:
    root = tk.Tk()
    game_gui = None
    match agent_name:
        case "heuristic":
            from src.ai.agent.heuristic_agent import HeuristicAgent
            game_gui = GameGUI(root, "2048 Game", HeuristicAgent())
        case "random":
            from src.ai.agent.random_agent import RandomAgent
            game_gui = GameGUI(root, "2048 Game", RandomAgent())
        case "expectimax":
            from src.ai.agent.expectimax_agent import ExpectimaxAgent
            game_gui = GameGUI(root, "2048 Game", ExpectimaxAgent())
    game_gui.run()

if __name__ == "__main__":
    # run("heuristic")
    # run("random")
    run("expectimax")
