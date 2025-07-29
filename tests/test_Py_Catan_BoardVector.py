import sys  
sys.path.append("../src/Py_Catan")
sys.path.append("./src")
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from Py_Catan.Preferences import PlayerPreferences
from Py_Catan.Player import Player
from Py_Catan.PlotBoard import PlotCatanBoard
from Py_Catan.Board import Board, BoardStructure, BoardLayout
from Py_Catan.PlayerValueFunctionBased import Player_Value_Function_Based
from Py_Catan.PlayerRandom import Player_Random 
from Py_Catan.PlayerPassive import Player_Passive
from Py_Catan.PlayerTestScenarios import TestPlayer
from Py_Catan.Tournament import Tournament
import Py_Catan.Player_Preference_Types as pppt
from Py_Catan.Trial import Trial
from Py_Catan.BoardVector import BoardVector
from Py_Catan.PlayerModelBased import Player_Model_Based

def test_board_vector_empty_board():
    '''
    Test the BoardVector class.
    Create a board, convert it to a vector, and then back to a board.
    Check if the original board and the new board are equal.
    '''
    # Create a board
    board = Board() 
    # Create a BoardVector from the board
    board_vector = BoardVector(board, include_values=True)
    # Convert the board to a vector
    vector = board_vector.create_vector_from_board()
    assert len(board_vector.header()) == board_vector.get_vector_size(), "Header size does not match vector size."
    assert len(board_vector.header()) == len(vector),"Header size does not match vector size."
    # Create a new board from the vector
    new_board = board_vector.create_board_from_vector()
    for p in new_board.players:
        assert all([n ==0 for n in p.villages]), "Not all village counts are zero."
        assert all([n ==0 for n in p.towns]), "Not all town counts are zero."
        assert all([n ==0 for n in p.streets]), "Not all street counts are zero."
        assert all([n ==0 for n in p.hand]), "Not all hand counts are zero."    
    # Convert the new board to a vector
    new_vector = board_vector.create_vector_from_board()
    # Check if the original vector and the new vector are equal 
    assert np.array_equal(vector, new_vector), "The original vector and the new vector are not equal."


def test_board_vector_villages():
    '''
    Test the BoardVector class.
    Build villages on the board.
    Create a board, convert it to a vector, and then back to a board.
    Check if the original board and the new board are equal.
    '''
    
    # Add villages to the board
    for player_to_build_village in range(4):
        # Create a board
        board = Board() 
        board.execute_player_action(player=board.players[player_to_build_village],best_action=('village', 10))
        board.players[player_to_build_village].hand = np.zeros(6, dtype=np.int32)  # Reset hand to zero
        # Create a BoardVector from the board
        board_vector = BoardVector(board, include_values=True)
        # Convert the board to a vector
        vector = board_vector.create_vector_from_board()
        index = board_vector.indices['nodes'][10]
        assert vector[index] == 1+player_to_build_village, "Village was not built correctly."
        # Create a new board from the vector
        new_board = board_vector.create_board_from_vector()
        for index,player in enumerate(new_board.players):
            if index == player_to_build_village:
                assert player.villages[10] == 1, "Village was not built correctly."
            else:
                assert all([n ==0 for n in player.villages]), "Not all village counts are zero."
            assert all([n ==0 for n in player.towns]), "Not all town counts are zero."
            assert all([n ==0 for n in player.streets]), "Not all street counts are zero."
            assert all([n ==0 for n in player.hand]), "Not all hand counts are zero."    
        # Convert the new board to a vector
        new_vector = board_vector.create_vector_from_board()
        # Check if the original vector and the new vector are equal 
        assert np.array_equal(vector, new_vector), "The original vector and the new vector are not equal."


def test_board_vector_towns():
    '''
    Test the BoardVector class.
    Build towns on the board.
    Create a board, convert it to a vector, and then back to a board.
    Check if the original board and the new board are equal.
    '''
    # Create a board
    
    # Add villages to the board
    for player_to_build_village in range(4):
        board = Board() 
        board.execute_player_action(player=board.players[player_to_build_village],best_action=('town', 10))
        board.players[player_to_build_village].hand = np.zeros(6, dtype=np.int32)  # Reset hand to zero
        # Create a BoardVector from the board
        board_vector = BoardVector(board, include_values=True)
        # Convert the board to a vector
        vector = board_vector.create_vector_from_board()
        index = board_vector.indices['nodes'][10]
        assert vector[index] == 5+player_to_build_village, "Town was not built correctly."
        # Create a new board from the vector
        new_board = board_vector.create_board_from_vector()
        for index,player in enumerate(new_board.players):
            if index == player_to_build_village:
                assert player.towns[10] == 1, "Town was not built correctly."
            else:
                assert all([n ==0 for n in player.towns]), "Not all town counts are zero."
            assert all([n ==0 for n in player.villages]), "Not all village counts are zero."
            assert all([n ==0 for n in player.streets]), "Not all street counts are zero."
            assert all([n ==0 for n in player.hand]), "Not all hand counts are zero."    
        # Convert the new board to a vector
        new_vector = board_vector.create_vector_from_board()
        # Check if the original vector and the new vector are equal 
        assert np.array_equal(vector, new_vector), "The original vector and the new vector are not equal."

def test_board_vector_streets():
    '''
    Test the BoardVector class.
    Build streets on the board.
    Create a board, convert it to a vector, and then back to a board.
    Check if the original board and the new board are equal.
    '''
    # Create a board

    # Add streets to the board
    for player_to_build_street in range(4):
        board = Board() 
        board.execute_player_action(player=board.players[player_to_build_street],best_action=('street', 10))
        board.players[player_to_build_street].hand = np.zeros(6, dtype=np.int32)  # Reset hand to zero
        # Create a BoardVector from the board
        board_vector = BoardVector(board, include_values=True)
        # Convert the board to a vector
        vector = board_vector.create_vector_from_board()
        index = board_vector.indices['edges'][10]
        assert vector[index] == 1+player_to_build_street, "Street was not built correctly."
        # Create a new board from the vector
        new_board = board_vector.create_board_from_vector()
        for index,player in enumerate(new_board.players):
            if index == player_to_build_street:
                assert player.streets[10] == 1, "Street was not built correctly."
            else:
                assert all([n ==0 for n in player.streets]), "Not all street counts are zero."
            assert all([n ==0 for n in player.villages]), "Not all village counts are zero."
            assert all([n ==0 for n in player.towns]), "Not all town counts are zero."
            assert all([n ==0 for n in player.hand]), "Not all hand counts are zero."    
        # Convert the new board to a vector
        new_vector = board_vector.create_vector_from_board()
        # Check if the original vector and the new vector are equal 
        assert np.array_equal(vector, new_vector), "The original vector and the new vector are not equal."

def test_board_vector_hands():
    '''
    Test the BoardVector class.
    add resources to the players' hands.
    Create a board, convert it to a vector, and then back to a board.
    Check if the original board and the new board are equal.
    '''
    for player_to_add_resources in range(4):
        # Create a board
        board = Board()
        board.players[player_to_add_resources].hand = np.array([1, 1, 1, 1, 1, 1], dtype=np.int32)
        # Create a BoardVector from the board
        board_vector = BoardVector(board, include_values=True)
        # Convert the board to a vector
        vector = board_vector.create_vector_from_board()
        indices = board_vector.indices['hand_for_player'][player_to_add_resources]
        assert all(vector[indices] == np.array([1, 1, 1, 1, 1, 1])), "Resource was not added correctly."
        # Create a new board from the vector
        new_board = board_vector.create_board_from_vector()
        for index,player in enumerate(new_board.players):
            if index == player_to_add_resources:
                assert all(player.hand == np.array([1, 1, 1, 1, 1, 1])), "Hand was not set correctly."
            else:
                assert all([n ==0 for n in player.hand]), "Not all hand counts are zero."
            assert all([n ==0 for n in player.villages]), "Not all village counts are zero."
            assert all([n ==0 for n in player.towns]), "Not all town counts are zero."
            assert all([n ==0 for n in player.streets]), "Not all street counts are zero."
        # Convert the new board to a vector
        new_vector = board_vector.create_vector_from_board()
        # Check if the original vector and the new vector are equal 
        assert np.array_equal(vector, new_vector), "The original vector and the new vector are not equal."

def test_board_vector_build_village_function():
    for player_to_build_village in range(4):
        # Create a board
        board = Board() 
        board.players[player_to_build_village].hand = np.array(board.structure.real_estate_cost[1])
        # Create a BoardVector from the board
        board_vector = BoardVector(board, include_values=True)
        # Convert the board to a vector
        vector = board_vector.build_village(player_to_build_village, 10)
        index = board_vector.indices['nodes'][10]
        assert vector[index] == 1+player_to_build_village, "Village was not built correctly."
        # Create a new board from the vector
        board_vector.vector = vector
        new_board = board_vector.create_board_from_vector()
        for index,player in enumerate(new_board.players):
            if index == player_to_build_village:
                assert player.villages[10] == 1, "Village was not built correctly."
            else:
                assert all([n ==0 for n in player.villages]), "Not all village counts are zero."
            assert all([n ==0 for n in player.towns]), "Not all town counts are zero."
            assert all([n ==0 for n in player.streets]), "Not all street counts are zero."
            assert all([n ==0 for n in player.hand]), "Not all hand counts are zero." 
            assert all([len(player.hand) == 6 for player in new_board.players]), "Not all players have a hand of length 6."
        # Convert the new board to a vector
        new_board_vector = BoardVector(new_board, include_values=False)
        new_vector = new_board_vector.vector
        # Check if the original vector and the new vector are equal 
        # exclude comparing on value, since after building in vector value has not been updated
        starting_index = max(board_vector.indices['values'])+1
        assert np.array_equal(vector[starting_index:], new_vector[starting_index:]), "The original vector and the new vector are not equal."

def test_board_vector_build_town_function():
    for player_to_build_town in range(4):
        # Create a board
        board = Board() 
        board.players[player_to_build_town].hand = np.array(board.structure.real_estate_cost[2])
        # Create a BoardVector from the board
        board_vector = BoardVector(board, include_values=True)
        # Convert the board to a vector
        vector = board_vector.build_town(player_to_build_town, 10)
        index = board_vector.indices['nodes'][10]
        assert vector[index] == 5+player_to_build_town, "Town was not built correctly."
        # Create a new board from the vector
        board_vector.vector = vector
        new_board = board_vector.create_board_from_vector()
        for index,player in enumerate(new_board.players):
            if index == player_to_build_town:
                assert player.towns[10] == 1, "Town was not built correctly."
            else:
                assert all([n ==0 for n in player.towns]), "Not all town counts are zero."
            assert all([n ==0 for n in player.villages]), "Not all village counts are zero."
            assert all([n ==0 for n in player.streets]), "Not all street counts are zero."
            assert all([n ==0 for n in player.hand]), "Not all hand counts are zero." 
            assert all([len(player.hand) == 6 for player in new_board.players]), "Not all players have a hand of length 6."
        # Convert the new board to a vector
        new_board_vector = BoardVector(new_board, include_values=False)
        new_vector = new_board_vector.vector
        # Check if the original vector and the new vector are equal 
        # exclude comparing on value, since after building in vector value has not been updated
        starting_index = max(board_vector.indices['values'])+1
        assert np.array_equal(vector[starting_index:], new_vector[starting_index:]), "The original vector and the new vector are not equal."


def test_board_vector_build_street_function():
    for player_to_build_street in range(4):
        # Create a board
        board = Board() 
        board.players[player_to_build_street].hand = np.array(board.structure.real_estate_cost[0])
        # Create a BoardVector from the board
        board_vector = BoardVector(board, include_values=True)
        # Convert the board to a vector
        vector = board_vector.build_street(player_to_build_street, 10)
        index = board_vector.indices['edges'][10]
        assert vector[index] == 1+player_to_build_street, "Street was not built correctly."
        # Create a new board from the vector
        board_vector.vector = vector
        new_board = board_vector.create_board_from_vector()
        for index,player in enumerate(new_board.players):
            if index == player_to_build_street:
                assert player.streets[10] == 1, "Street was not built correctly."
            else:
                assert all([n ==0 for n in player.streets]), "Not all street counts are zero."
            assert all([n ==0 for n in player.towns]), "Not all town counts are zero."
            assert all([n ==0 for n in player.villages]), "Not all street counts are zero."
            assert all([n ==0 for n in player.hand]), "Not all hand counts are zero." 
            assert all([len(player.hand) == 6 for player in new_board.players]), "Not all players have a hand of length 6."
        # Convert the new board to a vector
        new_board_vector = BoardVector(new_board, include_values=False)
        new_vector = new_board_vector.vector
        # Check if the original vector and the new vector are equal 
        # exclude comparing on value, since after building in vector value has not been updated
        starting_index = max(board_vector.indices['values'])+1
        assert np.array_equal(vector[starting_index:], new_vector[starting_index:]), "The original vector and the new vector are not equal."


def test_board_vector_trade_player_function():
    for player_to_trade in range(4):
        # Create a board
        board = Board() 
        board.players[player_to_trade].hand = np.array(board.structure.real_estate_cost[2])
        # Create a BoardVector from the board
        board_vector = BoardVector(board, include_values=True)
        # Convert the board to a vector
        vector = board_vector.trade_between_players(player_position = player_to_trade, card_out_in=(2,0))
        index_out = board_vector.indices['hand_for_player'][player_to_trade][2]
        index_in = board_vector.indices['hand_for_player'][player_to_trade][0]
        assert vector[index_out] == 1, "Card out not correct."
        assert vector[index_in] == 1, "Card in not correct."
        # Create a new board from the vector
        board_vector.vector = vector
        new_board = board_vector.create_board_from_vector()
        for index,player in enumerate(new_board.players):
            assert all([n ==0 for n in player.streets]), "Not all street counts are zero."
            assert all([n ==0 for n in player.towns]), "Not all town counts are zero."
            assert all([n ==0 for n in player.villages]), "Not all street counts are zero."
            if index == player_to_trade:
                assert player.hand[2] == 1, "Card out not correct."
                assert player.hand[0] == 1, "Card in not correct."
            else:
                assert all([n ==0 for n in player.hand]), "Not all hand counts are zero." 
            assert all([len(player.hand) == 6 for player in new_board.players]), "Not all players have a hand of length 6."
        # Convert the new board to a vector
        new_board_vector = BoardVector(new_board, include_values=False)
        new_vector = new_board_vector.vector
        # Check if the original vector and the new vector are equal 
        # exclude comparing on value, since after building in vector value has not been updated
        starting_index = max(board_vector.indices['values'])+1
        assert np.array_equal(vector[starting_index:], new_vector[starting_index:]), "The original vector and the new vector are not equal."



def test_board_vector_trade_bank_function():
    for player_to_trade in range(4):
        # Create a board
        board = Board() 
        board.players[player_to_trade].hand = np.array([4,0,0,0,0,0], dtype=np.int32)
        # Create a BoardVector from the board
        board_vector = BoardVector(board, include_values=True)
        # Convert the board to a vector
        vector = board_vector.trade_with_bank(player_position = player_to_trade, card_out_in=(0,3))
        index_out = board_vector.indices['hand_for_player'][player_to_trade][0]
        index_in = board_vector.indices['hand_for_player'][player_to_trade][3]
        assert vector[index_out] == 0, "Card out not correct."
        assert vector[index_in] == 1, "Card in not correct."
        # Create a new board from the vector
        board_vector.vector = vector
        new_board = board_vector.create_board_from_vector()
        for index,player in enumerate(new_board.players):
            assert all([n ==0 for n in player.streets]), "Not all street counts are zero."
            assert all([n ==0 for n in player.towns]), "Not all town counts are zero."
            assert all([n ==0 for n in player.villages]), "Not all street counts are zero."
            if index == player_to_trade:
                assert player.hand[0] == 0, "Card out not correct."
                assert player.hand[3] == 1, "Card in not correct."
            else:
                assert all([n ==0 for n in player.hand]), "Not all hand counts are zero." 
            assert all([len(player.hand) == 6 for player in new_board.players]), "Not all players have a hand of length 6."
        # Convert the new board to a vector
        new_board_vector = BoardVector(new_board, include_values=False)
        new_vector = new_board_vector.vector
        # Check if the original vector and the new vector are equal 
        # exclude comparing on value, since after building in vector value has not been updated
        starting_index = max(board_vector.indices['values'])+1
        assert np.array_equal(vector[starting_index:], new_vector[starting_index:]), "The original vector and the new vector are not equal."


def test_board_vector_multiple_actions():
    # Create a board
    board = Board() 
    board.players[0].hand = np.array(board.structure.real_estate_cost[0])
    board.execute_player_action(board.players[0], ('build_street', 0))
    board.players[1].hand = np.array(board.structure.real_estate_cost[0])
    board.execute_player_action(board.players[0], ('build_street', 10))
    board.players[2].hand = np.array(board.structure.real_estate_cost[1])
    board.execute_player_action(board.players[0], ('build_village', 0))
    board.players[3].hand = np.array(board.structure.real_estate_cost[1])
    board.execute_player_action(board.players[0], ('build_village', 20))
    board.players[3].hand = np.array(board.structure.real_estate_cost[2])
    board.execute_player_action(board.players[0], ('build_town', 20))
    board.players[0].hand = np.array(board.structure.real_estate_cost[0])
    board.players[1].hand = np.array(board.structure.real_estate_cost[0]) + np.array([0,0,0,0,0,1])  # Add a resource to player 1's hand
    board.players[2].hand = np.array([10,0,0,0,0,0])
    board.players[3].hand = np.array(board.structure.real_estate_cost[2]) + np.array(board.structure.real_estate_cost[1])
    # Create a BoardVector from the board
    original_board_vector = BoardVector(board, include_values=False) 
    # Build more streets, villages and towns
    board.execute_player_action(board.players[0], ('street', 20))
    board.execute_player_action(board.players[1], ('street', 30))
    board.execute_player_action(board.players[3], ('village', 30))
    board.execute_player_action(board.players[2], ('trade_bank', (0,2)))
    # Create a BoardVector from the board
    final_board_vector = BoardVector(board, include_values=False)
    # Same actions on original board vector
    original_board_vector.vector = original_board_vector.build_street(player_position=0, edge_index=20)
    original_board_vector.vector = original_board_vector.build_street(player_position=1, edge_index=30)
    original_board_vector.vector = original_board_vector.build_village(player_position=3, node_index=30)    
    original_board_vector.vector = original_board_vector.trade_with_bank(player_position=2,card_out_in=(0,2))
    # Check if the original vector and the new vector are equal 
    for index, pair in enumerate(zip(original_board_vector.vector, final_board_vector.vector)):
        if pair[0] != pair[1]:
            print(f"Vectors differ at index {index}: {pair[0]} != {pair[1]}")
            print(final_board_vector.header()[index])
    


    assert np.array_equal(final_board_vector.vector, original_board_vector.vector), "The original vector and the new vector are not equal."
