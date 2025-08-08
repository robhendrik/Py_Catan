from Py_Catan.Preferences import PlayerPreferences
from Py_Catan.BoardStructure import BoardStructure
from Py_Catan.Player import Player
from Py_Catan.Board import Board
from Py_Catan.BoardVector import BoardVector
from Py_Catan.PlayerValueFunctionBased import Player_Value_Function_Based
from Py_Catan.BoardLayout import BoardLayout
import Py_Catan.Player_Preference_Types as pppt
import numpy as np

def create_reference_board_1():
    # Create a board
    board_layout = BoardLayout(tile_layout='DSWOSBWWGSGSBGWOGOB')
    structure = BoardStructure(board_layout=board_layout)
    structure.winning_score = 8
    players = [ Player_Value_Function_Based(name = 'optimized_1',structure=structure,preference=pppt.optimized_1_with_0_for_full_score),
            Player_Value_Function_Based(name = 'optimized_2',structure=structure,preference=pppt.optimized_1),
            Player_Value_Function_Based(name = 'optimized_3',structure=structure,preference=pppt.optimized_2),
            Player_Value_Function_Based(name = 'optimized_4',structure=structure,preference=pppt.optimized_1)
        ]
    board = Board(structure=structure, players=players)
    #
    board.players[0].hand = np.array(board.structure.real_estate_cost[0])
    board.execute_player_action(board.players[0], ('street', 0))
    board.players[0].hand = np.array(board.structure.real_estate_cost[0])
    board.execute_player_action(board.players[0], ('street', 5))
    board.players[0].hand = np.array(board.structure.real_estate_cost[1])
    board.execute_player_action(board.players[0], ('village', 0))
    #
    for e in [15,16,17,18,19,20,21]:
        board.players[1].hand = np.array(board.structure.real_estate_cost[0])
        board.execute_player_action(board.players[1], ('street', e))
    #
    for n in [48,50,52]:
        board.players[2].hand = np.array(board.structure.real_estate_cost[1])
        board.execute_player_action(board.players[2], ('village', n))
    for e in [66,67,68,69]:
        board.players[2].hand = np.array(board.structure.real_estate_cost[0])
        board.execute_player_action(board.players[2], ('street', e))
    #
    board.players[3].hand = np.array(board.structure.real_estate_cost[1])
    board.execute_player_action(board.players[3], ('village', 20))
    board.players[3].hand = np.array(board.structure.real_estate_cost[0])
    board.execute_player_action(board.players[3], ('street', 25))
    board.players[3].hand = np.array(board.structure.real_estate_cost[1])
    board.execute_player_action(board.players[3], ('street', 24))
    board.players[3].hand = np.array(board.structure.real_estate_cost[2])
    board.execute_player_action(board.players[3], ('town', 20))
    #
    board.players[0].hand = np.array(board.structure.real_estate_cost[0])
    board.players[1].hand = np.array(board.structure.real_estate_cost[0]) + np.array([0,0,0,0,0,1])  # Add a resource to player 1's hand
    board.players[2].hand = np.array([10,0,0,0,0,0])
    board.players[3].hand = np.array(board.structure.real_estate_cost[2]) + np.array(board.structure.real_estate_cost[1])
    #
    board._update_board_for_players()
    for p in board.players:
        p.update_build_options()
    return board