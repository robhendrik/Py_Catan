import sys  
sys.path.append("./src/Py_Catan")
sys.path.append("./src")
import pytest
import numpy as np

from Py_Catan.BoardStructure import BoardStructure, BoardLayout
from Py_Catan.PlotBoard import PlotCatanBoard
from Py_Catan.Preferences import PlayerPreferences
from Py_Catan.Player import Player
from Py_Catan.Board import Board
from Py_Catan.PlayerValueFunctionBased import Player_Value_Function_Based
from Py_Catan.PlayerModelBased import Player_Model_Based
from keras.models import Model
from keras.saving import load_model

test_model_path = './tests/test_models/test_model_wth_jul_27_dataset.keras'

def test_respond_to_trading_request():
    """
    Test if the player responds to a trading request.
    """
    board_layout = BoardLayout(tile_layout='DSWOSBWWGSGSBGWOGOB')
    structure = BoardStructure(board_layout=board_layout)
    model = load_model(test_model_path, safe_mode=False)
    p = Player_Model_Based(name='Player_5', structure=structure, model=model)
    brd = Board(structure=structure)
    brd.players[3] = p
    brd._update_board_for_players()
    p.hand = np.array([1, 0, 1, 0, 0, 0])
    assert False == p.respond_positive_to_other_players_trading_request(card_out_in=(0,2))
    assert True == p.respond_positive_to_other_players_trading_request(card_out_in=(2,0))

def test_generate_list_of_possible_actions():
    """
    Test the generate_list_of_possible_actions method of Player_Model_Based.
    Checks if the returned actions are correct and complete for a board with some buildings.
    """
    # Setup: create a board and player
    board_layout = BoardLayout(tile_layout='DSWOSBWWGSGSBGWOGOB')
    structure = BoardStructure(board_layout=board_layout)
    model = load_model(test_model_path, safe_mode=False)
    player = Player_Model_Based(name='Player_Test', structure=structure, model=model)
    brd = Board(structure=structure)
    brd.players[3] = player
    brd._update_board_for_players()
    
    # Simulate some buildings for the player
    # For example, build a village at position 0 and a street at position 0-1
    brd.execute_player_action(player, ('village', 9))
    brd.execute_player_action(player, ('village', 11))
    brd.execute_player_action(player, ('street', 15))
    brd.execute_player_action(player, ('street', 16))
    brd.execute_player_action(player, ('street', 33))

    # easy case
    player.hand = np.array([0, 0, 0, 0, 0, 0])  # empty hand for simplicity
    brd._update_board_for_players()
    player.update_build_options()
    actions = player.generate_list_of_possible_actions(rejected_trades=dict([]))
    assert len(actions) == 0, "Expected no actions possible"

    # street
    player.hand = np.array([1, 0, 0, 0, 0, 1])  # can build a street
    brd._update_board_for_players()
    player.update_build_options()
    actions = player.generate_list_of_possible_actions(rejected_trades=dict([]))
    possible_streets = [8,14,17,32,48,49]
    for s in possible_streets:
        assert ('street', s) in actions, f"Expected action ('street', {s}) to be possible"
    assert 6 == sum(1 for action in actions if action[0] == 'street')
    
    # town
    player.hand = np.array(brd.structure.real_estate_cost[2])  # can build a town
    brd._update_board_for_players()
    player.update_build_options()
    actions = player.generate_list_of_possible_actions(rejected_trades=dict([]))
    possible_towns = [9,11]
    for t in possible_towns:
        assert ('town', t) in actions, f"Expected action ('town', {t}) to be possible"
    assert 2 == sum(1 for action in actions if action[0] == 'town')

    # village
    player.hand = np.array(brd.structure.real_estate_cost[1])  # can build a village
    brd._update_board_for_players()
    player.update_build_options()
    actions = player.generate_list_of_possible_actions(rejected_trades=dict([]))
    possible_villages = [31]
    for v in possible_villages:
        assert ('village', v) in actions, f"Expected action ('village', {v}) to be possible"
    assert 1 == sum(1 for action in actions if action[0] == 'village')

def test_generate_list_of_values():
    """
    Test the generate_list_of_possible_actions method of Player_Model_Based.
    Checks if the returned actions are correct and complete for a board with some buildings.
    """
    # Setup: create a board and player
    board_layout = BoardLayout(tile_layout='DSWOSBWWGSGSBGWOGOB')
    structure = BoardStructure(board_layout=board_layout)
    model = load_model(test_model_path, safe_mode=False)
    player = Player_Model_Based(name='Player_Test', structure=structure, model=model)
    brd = Board(structure=structure)
    brd.players[3] = player
    brd._update_board_for_players()
    
    # Simulate some buildings for the player
    # For example, build a village at position 0 and a street at position 0-1
    brd.execute_player_action(player, ('village', 9))
    brd.execute_player_action(player, ('village', 11))
    brd.execute_player_action(player, ('street', 15))
    brd.execute_player_action(player, ('street', 16))
    brd.execute_player_action(player, ('street', 33))

    # easy case
    player.hand = np.array([1, 0, 0, 0, 0, 0])  # only trading for simplicity
    brd._update_board_for_players()
    player.update_build_options()
    actions = player.generate_list_of_possible_actions(rejected_trades=dict([]))
    values = player.generate_values_for_possible_actions(actions)
    assert len(values) == 5
    
    # street
    player.hand = np.array(brd.structure.real_estate_cost[0])  # can build a street
    brd._update_board_for_players()
    player.update_build_options()
    actions = player.generate_list_of_possible_actions(rejected_trades=dict([]))
    values = player.generate_values_for_possible_actions(actions)
    assert len(values) == len(actions)

def test_best_action_for_fourth_player_who_likes_B_cards():
    """
    test if a model based player with preference for B-cards indeed trades for B-cards
    """
    # Setup: create a board and player
    board_layout = BoardLayout(tile_layout='DSWOSBWWGSGSBGWOGOB')
    structure = BoardStructure(board_layout=board_layout)
    model = load_model(test_model_path, safe_mode=False)
    player = Player_Model_Based(name='Player_Test', structure=structure, model=model)
    brd = Board(structure=structure)
    brd.players[3] = player # this players likes B -cards
    brd._update_board_for_players()
    
    # Simulate some buildings for the player
    # For example, build a village at position 0 and a street at position 0-1
    brd.execute_player_action(player, ('village', 9))
    brd.execute_player_action(player, ('village', 11))
    brd.execute_player_action(player, ('street', 15))
    brd.execute_player_action(player, ('street', 16))
    brd.execute_player_action(player, ('street', 33))

    # easy case
    player.hand = np.array([0, 0, 1, 0, 0, 0])  # only trading for simplicity
    brd._update_board_for_players()
    player.update_build_options()
    actions = player.generate_list_of_possible_actions(rejected_trades=dict([]))
    values = player.generate_values_for_possible_actions(actions)
    best_action = player.find_best_action(rejected_trades_for_this_round=dict([]))
    assert len(values) == 5
    assert best_action  == ('trade_player', (2, 0))
    # street
    player.hand = np.array(brd.structure.real_estate_cost[0])  # can build a street
    brd._update_board_for_players()
    player.update_build_options()
    actions = player.generate_list_of_possible_actions(rejected_trades=dict([]))
    values = player.generate_values_for_possible_actions(actions)
    
    best_action = player.find_best_action(rejected_trades_for_this_round=dict([]))

    assert len(values) == len(actions)
    assert best_action[1][1]  == 0 # card_in is card 0, the preferred B-card
    assert best_action[0] == 'trade_player'


def test_best_action_for_first_player_who_likes_streets():
    """
    test if a model based player with preference for streets indeed builds streets
    """
    # Setup: create a board and player
    board_layout = BoardLayout(tile_layout='DSWOSBWWGSGSBGWOGOB')
    structure = BoardStructure(board_layout=board_layout)
    model = load_model(test_model_path, safe_mode=False)
    player = Player_Model_Based(name='Player_Test', structure=structure, model=model)
    brd = Board(structure=structure)
    brd.players[0] = player # this players likes streets
    brd._update_board_for_players()
    
    # Simulate some buildings for the player
    # For example, build a village at position 0 and a street at position 0-1
    brd.execute_player_action(player, ('village', 9))
    brd.execute_player_action(player, ('village', 11))
    brd.execute_player_action(player, ('street', 15))
    brd.execute_player_action(player, ('street', 16))
    brd.execute_player_action(player, ('street', 33))

    # easy case
    player.hand = np.array([0, 0, 1, 0, 0, 0])  # only trading for simplicity
    brd._update_board_for_players()
    player.update_build_options()
    actions = player.generate_list_of_possible_actions(rejected_trades=dict([]))
    values = player.generate_values_for_possible_actions(actions)
    best_action = player.find_best_action(rejected_trades_for_this_round=dict([]))
    assert len(values) == 5
    assert best_action[0]  == 'trade_player'
    # street
    player.hand = np.array(brd.structure.real_estate_cost[0])  # can build a street
    brd._update_board_for_players()
    player.update_build_options()
    actions = player.generate_list_of_possible_actions(rejected_trades=dict([]))
    values = player.generate_values_for_possible_actions(actions)
    best_action = player.find_best_action(rejected_trades_for_this_round=dict([]))
    assert len(values) == len(actions)
    assert best_action[0] == 'street'
    # all options possible
    player.hand = np.array([2, 2, 2, 2, 2, 2])  # can build anything
    brd._update_board_for_players()
    player.update_build_options()
    actions = player.generate_list_of_possible_actions(rejected_trades=dict([]))
    values = player.generate_values_for_possible_actions(actions)
    best_action = player.find_best_action(rejected_trades_for_this_round=dict([]))
    assert len(values) == len(actions)
    assert best_action[0] == 'street'

def test_best_action_for_second_player_who_likes_villages():
    """
    test if a model based player with preference for villages indeed builds villages
    """
    # Setup: create a board and player
    board_layout = BoardLayout(tile_layout='DSWOSBWWGSGSBGWOGOB')
    structure = BoardStructure(board_layout=board_layout)
    model = load_model(test_model_path, safe_mode=False)
    player = Player_Model_Based(name='Player_Test', structure=structure, model=model)
    brd = Board(structure=structure)
    brd.players[1] = player # this players likes villages
    brd._update_board_for_players()
    
    # Simulate some buildings for the player
    # For example, build a village at position 0 and a street at position 0-1
    brd.execute_player_action(player, ('village', 9))
    brd.execute_player_action(player, ('village', 11))
    brd.execute_player_action(player, ('street', 15))
    brd.execute_player_action(player, ('street', 16))
    brd.execute_player_action(player, ('street', 33))

    # easy case
    player.hand = np.array([0, 0, 1, 0, 0, 0])  # only trading for simplicity
    brd._update_board_for_players()
    player.update_build_options()
    actions = player.generate_list_of_possible_actions(rejected_trades=dict([]))
    values = player.generate_values_for_possible_actions(actions)
    best_action = player.find_best_action(rejected_trades_for_this_round=dict([]))
    assert len(values) == 5
    assert best_action[0]  == 'trade_player'
    # village
    player.hand = np.array(brd.structure.real_estate_cost[1])  # can build a village
    brd._update_board_for_players()
    player.update_build_options()
    actions = player.generate_list_of_possible_actions(rejected_trades=dict([]))
    values = player.generate_values_for_possible_actions(actions)
    best_action = player.find_best_action(rejected_trades_for_this_round=dict([]))
    assert len(values) == len(actions)
    assert best_action[0] == 'village'
    # all options possible
    player.hand = np.array([3,3,3,3,3,3])  # can build anything
    brd._update_board_for_players()
    player.update_build_options()
    actions = player.generate_list_of_possible_actions(rejected_trades=dict([]))
    values = player.generate_values_for_possible_actions(actions)
    best_action = player.find_best_action(rejected_trades_for_this_round=dict([]))
    assert len(values) == len(actions)
    assert best_action[0] == 'village'


def test_best_action_for_third_player_who_likes_towns():
    """
    test if a model based player with preference for towns indeed builds towns
    """

    # Setup: create a board and player
    board_layout = BoardLayout(tile_layout='DSWOSBWWGSGSBGWOGOB')
    structure = BoardStructure(board_layout=board_layout)
    model = load_model(test_model_path, safe_mode=False)
    player = Player_Model_Based(name='Player_Test', structure=structure, model=model)
    brd = Board(structure=structure)
    brd.players[2] = player # this players likes towns
    brd._update_board_for_players()
    
    # Simulate some buildings for the player
    # For example, build a village at position 0 and a street at position 0-1
    brd.execute_player_action(player, ('village', 9))
    brd.execute_player_action(player, ('village', 11))
    brd.execute_player_action(player, ('street', 15))
    brd.execute_player_action(player, ('street', 16))
    brd.execute_player_action(player, ('street', 33))

    # easy case
    player.hand = np.array([0, 0, 1, 0, 0, 0])  # only trading for simplicity
    brd._update_board_for_players()
    player.update_build_options()
    actions = player.generate_list_of_possible_actions(rejected_trades=dict([]))
    values = player.generate_values_for_possible_actions(actions)
    best_action = player.find_best_action(rejected_trades_for_this_round=dict([]))
    assert len(values) == 5
    assert best_action[0]  == 'trade_player'
    # town
    player.hand = np.array(brd.structure.real_estate_cost[2])  # can build a town
    brd._update_board_for_players()
    player.update_build_options()
    actions = player.generate_list_of_possible_actions(rejected_trades=dict([]))
    values = player.generate_values_for_possible_actions(actions)
    best_action = player.find_best_action(rejected_trades_for_this_round=dict([]))
    assert len(values) == len(actions)
    assert best_action[0] == 'town'
    # all options possible
    player.hand = np.array([3,3,3,3,3,3])  # can build anything
    brd._update_board_for_players()
    player.update_build_options()
    actions = player.generate_list_of_possible_actions(rejected_trades=dict([]))
    values = player.generate_values_for_possible_actions(actions)
    best_action = player.find_best_action(rejected_trades_for_this_round=dict([]))
    assert len(values) == len(actions)
    assert best_action[0] == 'town'