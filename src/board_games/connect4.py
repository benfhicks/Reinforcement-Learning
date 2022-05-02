import numpy as np
from random import randint


class Game:
    def __init__(self) -> None:
        self.board = self._create_board()
        self.history = []

    def _create_board(shape: tuple = (6, 6)):
        board = np.zeros(shape=shape, dtype=int)

        return board

    def clear_board(self):
        self.board = self._create_board()

    def display_board(self) -> str:
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                counter = " "
                if self.board[i][j] == 1:
                    counter = "X"
                elif self.board[i][j] == 2:
                    counter = "O"

                if j == len(self.board[i]) - 1:
                    print(str(counter) + "|", end="")
                if j == 0:
                    print("|" + str(counter), end="")
                else:
                    print(str(counter))

            if i == 0:
                print("-" * (2 + len(self.board[i])))

    def _valid_moves(self) -> np.array:
        pass

    def make_move(self, player: int, move: np.array):
        pass

    def check_winner(self):
        pass

    def _simulate_random_game(self):
        pass
