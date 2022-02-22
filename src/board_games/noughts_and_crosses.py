import numpy as np
from random import randint

class Game():
    def __init__(self) -> None:
        self.board = self._create_board()
        self.history = []

    def _create_board(self) -> np.array:
        board = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ])
        return board

    def clear_board(self) -> np.array:
        self.board = self._create_board(self)

    def _valid_moves(self) -> np.array:
        moves = np.argwhere(self.board == 0)
        return moves

    def make_move(self, player: int, move: np.array):
        self.history.append([player, (move[0], move[1])])
        self.board[move[0], move[1]] = player

    def _check_winner(self) -> int:
        """
        Here we define a function to check whether or not we have a winner.

        This function has 4 states:
            -1: still in progress
            0: draw
            1: player 1 wins
            2: player 2 wins
        """
        if self._check_rows() != 0:
            print('rows')
            return self._check_rows()
        elif self._check_columns() != 0:
            print('cols')
            return self._check_columns()
        elif self._check_diagonals() != 0:
            print('diag')
            return self._check_diagonals()
        elif len(self._valid_moves()) == 0:
            return 0
        else:
            return -1

    def _check_rows(self) -> int:
        for row in self.board:
            if len(set(row)) == 1:
                return row[0]
        return 0

    def _check_columns(self) -> int:
        board_transpose = self.board.T
        for row in board_transpose:
            if len(set(row)) == 1:
                return row[0]
        return 0

    def _check_diagonals(self) -> int:
        if len(set([self.board[i][i] for i in range(len(self.board))])) == 1:
            return self.board[0][0]
        if len(set([self.board[i][len(self.board)-1-i] for i in range(len(self.board))])) == 1:
            return self.board[0][len(self.board)-1]
        return 0

    def display_board(self) -> str:
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                counter = ' '
                if self.board[i][j] == 1:
                    counter = 'X'
                elif self.board[i][j] == 2:
                    counter  = 'O'

                if (j == len(self.board[i]) - 1):
                    print(counter)
                else:
                    print(str(counter) + "|", end='')
                    
            if (i < len(self.board) - 1):
                print("-----")

    def _simulate_random_game(self):
        history = []
        player = 1

        while self._check_winner() == -1:
            possible_moves = self._valid_moves()
            move_choice = possible_moves[randint(0, len(possible_moves)-1)]

            move = self.make_move(player, move_choice)

            player = 1 if player == 2 else 2

        self.display_board()

    def _get_moves(self):
        return [x[1] for x in self.history]