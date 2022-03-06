import numpy as np
from random import randint
import torch
from torch import nn
from torch.utils.data import Dataset

class Game():
    def __init__(self) -> None:
        self.board = self._create_board()
        self.history = []

    def _create_board(self) -> np.array:
        board = np.zeros(shape = (3,3), dtype = int)
        
        return board

    def clear_board(self):
        self.board = self._create_board()
        
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

    def _valid_moves(self) -> np.array:
        moves = np.argwhere(self.board == 0)
        return moves

    def make_move(self, player: int, move: np.array):
        self.board[move[0], move[1]] = player

    def _history_to_board_states(self):
        board_states = []
        winners = []
        
        for game in self.history:
            board = self._create_board()
            winner = game[1]
            for move in game[0]:
                player = move[0]
                move_played = move[1]
                board[move_played[0], move_played[1]] = player
                
                board_states.append(board.copy())
                winners.append(winner)

        return board_states, winners
        
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
            return self._check_rows()
        elif self._check_columns() != 0:
            return self._check_columns()
        elif self._check_diagonals() != 0:
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

    def _simulate_random_game(self, display: bool = True, n: int = 1):
        for i in range(n):
            self.clear_board()
            move_history = []
            player = 1

            while self._check_winner() == -1:
                possible_moves = self._valid_moves()
                move_choice = possible_moves[randint(0, len(possible_moves)-1)]
                self.make_move(player, move_choice)
                move_history.append((player, move_choice))
                player = 1 if player == 2 else 2

            if n == 1 and display:
                self.display_board()
            
            self.history.append([move_history, self._check_winner()])
            
    def _simulate_game(self, display: bool = True, n: int = 1, p1 = None, p2 = None):
        for i in range(n):
            self.clear_board()
            move_history = []
            player = 1

            while self._check_winner() == -1:
                possible_moves = self._valid_moves()
                model = p1 if p1 is not None else p2
                
                if player == 1 and p1 is None:
                    move_choice = possible_moves[randint(0, len(possible_moves)-1)]
                elif player == 2 and p2 is None:
                    move_choice = possible_moves[randint(0, len(possible_moves)-1)]
                else:
                    move_choice = self._best_move(player = player, model = model)
                
                self.make_move(player, move_choice)
                move_history.append((player, move_choice))
                player = 1 if player == 2 else 2

            if n == 1 and display:
                self.display_board()
            
            self.history.append([move_history, self._check_winner()])
            
    def game_statistics(self):
        winners = []
        
        for game in self.history:
            winners.append(game[1])
        
        winner, wins =  np.unique(winners, return_counts=True)
        outcomes = dict(zip(winner, wins))
        
        
        player_1_wins = outcomes[1] if 1 in outcomes else 0
        player_2_wins = outcomes[2] if 2 in outcomes else 0
        draws = outcomes[0] if 0 in outcomes else 0
        total_games = player_1_wins + player_2_wins + draws

        print(f"""--------------------------------
             \nTotal games:\t{total_games}
             \n--------------------------------
             \nPlayer 1 wins:\t{player_1_wins}\t({(player_1_wins / total_games)*100:.2f}%)
             \nPlayer 2 wins:\t{player_2_wins}\t({(player_2_wins / total_games)*100:.2f}%)
             \nDraws:\t\t{draws}\t({(draws / total_games)*100:.2f}%)
             \n--------------------------------""")

    def _board_to_tensor(self, board):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.tensor(np.asarray(board, dtype='float32').reshape(-1,9)).to(device)

    def _best_move(self, player: int, model = None):
        move_scores = []
        possible_moves = self._valid_moves()

        if model is None:
            return possible_moves[randint(0, len(possible_moves)-1)]
        else:
            for move in possible_moves:
                temp_board = self.board.copy()
                temp_board[move[0], move[1]] = player
                temp_board = self._board_to_tensor(temp_board)
                
                model_output = model(temp_board)

                draw_prediction = model_output[0][0].item()
                win_prediction = model_output[0][1].item() if player == 1 else model_output[2]
                lose_prediction = model_output[0][2].item() if player == 1 else model_output[1]
                
                move_scores.append(win_prediction if win_prediction > draw_prediction else draw_prediction)
                ## Now we need to find the best move out of our list of moves
            return possible_moves[np.argmax(move_scores)]
        
    def _get_moves(self):
        return [x[1] for x in self.history]
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(9, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        return x
    
class GameResults(Dataset):
    def __init__(self, board_states, winners):
        self.board_state = np.asarray(board_states, dtype='float32')
        self.outcome_state = winners

    def __len__(self):
        return len(self.outcome_state)

    def __getitem__(self, idx):
        board_state = self.board_state[idx]
        outcome_state = self.outcome_state[idx]
        sample = {"Board state": board_state, "Outcome state": outcome_state}
        return sample