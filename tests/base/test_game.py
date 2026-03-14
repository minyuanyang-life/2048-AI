from src.core.game import Game
from src.core.enums import GameState, Direction

def main():
    game = Game()
    direction = None
    while True:
        print("\n"*10)
        print(game)
        c = input()
        match c:
            case "a": direction = Direction.LEFT
            case "A": direction = Direction.LEFT
            case "d": direction = Direction.RIGHT
            case "D": direction = Direction.RIGHT
            case "w": direction = Direction.UP
            case "W": direction = Direction.UP
            case "s": direction = Direction.DOWN
            case "S": direction = Direction.DOWN
            case "q": exit()
            case "Q": exit()
            case  _ : continue

        result, new_game = game.step(direction)
        if result == GameState.ERROR:
            print("Unknown Error")
            exit()

        if result == GameState.OVER:
            print("\n"*10)
            print(game)
            print("Game Over")
            while True:
                pass

        game = new_game


if __name__ == "__main__":
    main()