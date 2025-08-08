from Py_Catan.Preferences import PlayerPreferences
from Py_Catan.BoardStructure import BoardStructure
from Py_Catan.Player import Player
from Py_Catan.Board import Board
from Py_Catan.BoardVector import BoardVector
from Py_Catan.ValueFunction import ValueFunction
import numpy as np


def value_for_board_vector(board_vector: BoardVector) -> np.ndarray:
    board = board_vector.create_board_from_vector(list_of_players=board_vector.players)
    return value_for_board(board)


def value_for_board(board = Board) -> np.ndarray:
    board._update_board_for_players()
    values = []
    for player in board.players:
        player.update_build_options()
        values.append(player.calculate_value())
    return np.array(values,np.float32)


def board_vector_from_vector(vector: np.ndarray, structure: BoardStructure, players: list[Player]) -> np.ndarray:
    board = Board(structure=structure, players=players)
    board_vector = BoardVector(board=board, include_values = False) 
    board_vector.vector = vector
    return board_vector