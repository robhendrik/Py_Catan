import sys  
sys.path.append("./src/Py_Catan")
sys.path.append("./src")
import pytest
import numpy as np

#from Py_Catan.GenBoard import GenBoard
from Py_Catan.BoardStructure import BoardStructure
from Py_Catan.Preferences import PlayerPreferences
from Py_Catan.Player import Player
from Py_Catan.Board import Board
from Py_Catan.PlayerRandom import Player_Random
from Py_Catan.PlayerPassive import Player_Passive
from Py_Catan.BoardStructure import BoardLayout
from Py_Catan.Tournament import Tournament

def test_random_player():
    board_layout = BoardLayout(tile_layout='DSWOSBWWGSGSBGWOGOB')
    structure = BoardStructure(board_layout=board_layout)
    structure.winning_score = 8
    brd = Board(structure=structure) 
    players = []
    for name in ['A','B','C','D']:
        players.append(Player_Random(name = name, structure = structure ))
    brd.players = players
    for p in brd.players + brd.players[::-1]:
        actions = p.player_setup(brd)
        brd.execute_player_action(p, actions[0])
        brd.execute_player_action(p, actions[1])
    for p in brd.players:
        assert sum(p.streets) == 2
        assert sum(p.villages) == 2
        assert sum(p.towns) == 0
        assert sum(p.hand) == 0

    board_layout = BoardLayout(tile_layout='DSWOSBWWGSGSBGWOGOB')
    structure = BoardStructure(board_layout=board_layout)
    structure.winning_score = 8
    brd = Board(structure=structure) 
    players = []
    for name in ['A','B','C','D']:
        players.append(Player_Random(name = name, structure = structure ))
    brd.players = players
    p = players[0]
    p.hand = np.array(brd.structure.real_estate_cost[0])
    brd.execute_player_action(p,('street',29))
    p.hand = np.array(brd.structure.real_estate_cost[1])
    brd.execute_player_action(p,('village',6))
    p.hand = np.array(brd.structure.real_estate_cost[0])
    p.update_build_options()
    for _ in range(10):
        best_action = p.find_best_action(rejected_trades_for_this_round=dict([]))
        assert best_action[0] in ['street','trade_player',None]
        if best_action[0] == 'street':
            assert best_action[1] in [30,12,28,6]    
        if best_action[0] == 'trade_player':
            assert best_action[1][0] in [0,5]
    for _ in range(10):
        best_action = p.find_best_action(rejected_trades_for_this_round=dict([]))
        assert best_action[0] in ['street','trade_player']
        if best_action[0] == 'street':
            assert best_action[1] in [30,12,28,6]    
        if best_action[0] == 'trade_player':
            assert best_action[1][0] in [0,5]
    p.threshold_for_accepting_trade = -0.1
    assert p.respond_positive_to_other_players_trading_request(card_out_in=(0,5)) == False
    p.threshold_for_accepting_trade = 1.1
    assert p.respond_positive_to_other_players_trading_request(card_out_in=(0,5)) == True

def test_random_player_number_of_action():
    board_layout = BoardLayout(tile_layout='DSWOSBWWGSGSBGWOGOB')
    structure = BoardStructure(board_layout=board_layout)
    structure.winning_score = 8
    brd = Board(structure=structure) 
    tournament = Tournament()
    players = []
    for name in ['A','B','C','D']:
        players.append(Player_Random(name = name, structure = structure ))
    brd.players = players
    assert all([player._actions_in_round == 0 for player in brd.players])
    p = players[0]
    p.max_actions_in_round = 5
    # build one street
    p.hand = np.array(brd.structure.real_estate_cost[0])
    brd.execute_player_action(p,('street',29))
    p.update_build_options()
    # with hand for street the player should be able to build a street
    p.hand = np.array(brd.structure.real_estate_cost[0])
    best_action = p.find_best_action(rejected_trades_for_this_round=dict([]))
    assert best_action[0] == 'street'
    # for 5 actions in a turn the player should be able to build 5 streets,
    # so total number of streets should be 6
    for _ in range(5):
        p.hand = np.array(brd.structure.real_estate_cost[0])
        tournament.player_action_in_turn(brd, p, rejected_trades_for_this_round=dict([]))
    assert p._actions_in_round == 5
    assert sum(p.streets) == 6
    # since we did set max_actions_in_round to 5, the player should not be able to build more streets
    # so the best action should be None
    p.hand = np.array(brd.structure.real_estate_cost[0])
    best_action = p.find_best_action(rejected_trades_for_this_round=dict([]))
    assert best_action[0] == None
    # reset actions in round by throwing dice (new turn)
    brd.throw_dice()
    p.hand = np.array(brd.structure.real_estate_cost[0])
    best_action = p.find_best_action(rejected_trades_for_this_round=dict([]))
    assert best_action[0] == 'street'
    # if cannot build, proposal should be to trade, force trade with bank
    p.hand = np.array([5,0,0,0,0,0])
    best_action = p.find_best_action(rejected_trades_for_this_round=set({(0,5)}))
    assert best_action[0] == 'trade_player'
    best_action = p.find_best_action(rejected_trades_for_this_round=set({(0,1),(0,2),(0,3),(0,4),(0,5)}))
    assert best_action[0] == 'trade_bank'