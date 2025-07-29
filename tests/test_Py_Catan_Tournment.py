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

def test_calculate_points():
    """
    Test the calculate_points function to ensure it correctly calculates the points for each player.
    """
    board_layout = BoardLayout(tile_layout='DSWOSBWWGSGSBGWOGOB')
    structure = BoardStructure(board_layout=board_layout)
    structure.winning_score = 8

    players = [
        Player_Value_Function_Based(name='Player1', structure=structure, preference=pppt.strong_1),
        Player_Value_Function_Based(name='Player2', structure=structure, preference=pppt.strong_2),
        Player_Value_Function_Based(name='Player3', structure=structure, preference=pppt.strong_1),
        Player_Passive(name='Player4', structure=structure)
    ]

    tournament = Tournament()
    tournament.score_table_for_ranking_per_game = [10, 5, 2, 0]
    res = tournament.calculate_points([1,2,3,4])
    assert all(np.array(res) == np.array([0,2,5,10]))
    res = tournament.calculate_points([1,2,3,3])
    assert all(np.array(res) == np.array([0,2,7.5,7.5]))
    res = tournament.calculate_points([12,2,2,1])
    assert all(np.array(res) == np.array([10,3.5,3.5,0]))
    res = tournament.calculate_points([2,2,2,2])
    assert all(np.array(res) == np.array([4.25,4.25,4.25,4.25]))

def test_order():
    input_list = ['a', 'b', 'c', 'd']
    tournament = Tournament()   
    for n in range(24):
        ordered_list = tournament._order_elements(n, input_list)
        original_list = tournament._order_elements(n, ordered_list, reverse=True)
        assert original_list == input_list
    assert len(tournament.list_of_orders) == 24
   
def test_game():
    """
    Test the game functionality with a custom player that takes specific actions based on its position in the turn order.
    """
    board_layout = BoardLayout(tile_layout='DSWOSBWWGSGSBGWOGOB')
    structure = BoardStructure(board_layout=board_layout)
    structure.winning_score = 8

    players = [
        TestPlayer(name='Player1', structure=structure),
        TestPlayer(name='Player2', structure=structure),
        TestPlayer(name='Player3', structure=structure),
        TestPlayer(name='Player4', structure=structure)
    ]
    for player in players:
        player._players_in_this_game = players

    tournament = Tournament()
    results, rounds = tournament.play_game(board_structure=structure, players=players)
    assert len(results) == 4
    assert all(r == 4 for r in rounds)
    assert all(results == np.array([8,4,2,0]))

def test_tournament():
    """
    Test the tournament functionality with a set of players and verify the results.
    """
    board_layout = BoardLayout(tile_layout='DSWOSBWWGSGSBGWOGOB')
    structure = BoardStructure(board_layout=board_layout)
    structure.winning_score = 8

    players = [
        TestPlayer(name='Player1', structure=structure),
        TestPlayer(name='Player2', structure=structure),
        TestPlayer(name='Player3', structure=structure),
        TestPlayer(name='Player4', structure=structure)
    ]
    for player in players:
        player._players_in_this_game = players

    tournament = Tournament()
    tournament.list_of_orders = [[0,1,2,3],[1,0,2,3]]
    tournament.list_of_reversed_orders = tournament._create_list_of_reversed_orders()
    tournament.score_table_for_ranking_per_game = [10, 5, 2, 0]
    tournament.no_games_in_tournament = 2
    tournament.verbose = False
    # We play two game with the test players. In first "Player1" scores 8 points and 10 victory points in 4 rounds,
    # while "Player2" scores 4 points and 5 victory points in 4 rounds. Player3 scores 2 points and 2 victory points.
    # in Second game "Player1" and "Player2" are reversed
    # so we expect for results [12,12,4,0], for victory points [15,15,4,0] and for rounds [4,4,4,4]
    player_tournament_results, player_victory_points, rounds_for_this_game = tournament.tournament(structure, players)
    
    assert all(player_tournament_results == np.array([12,12,4,0]))
    assert all(player_victory_points == np.array([15,15,4,0]))
    assert all(rounds_for_this_game == np.array([4,4,4,4]))

def test_value_player_number_of_action():
    board_layout = BoardLayout(tile_layout='DSWOSBWWGSGSBGWOGOB')
    structure = BoardStructure(board_layout=board_layout)
    structure.winning_score = 8
    brd = Board(structure=structure) 
    tournament = Tournament()
    players = []
    for name in ['A','B','C','D']:
        # create players who like streets, and 'hands' for streets. They do not like a hand for a village or town
        pref = PlayerPreferences()
        pref.villages, pref.towns, pref.streets, pref.cards_in_hand, pref.cards_earning_power = 1,1,10,0,0
        pref.hand_for_street = 1
        pref.hand_for_village = 0
        pref.hand_for_town = 0
        pref.penalty_reference_for_too_many_cards = 1
        pref.resource_type_weight = np.array([1,0,1,1,1,1])
        pref = pref.normalized()
        players.append(Player_Value_Function_Based(preference=pref))
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
    p.hand = np.array([2,1,1,1,1,2])
    best_action = p.find_best_action(rejected_trades_for_this_round=dict([]))
    assert best_action[0] == 'street'
    # for 5 actions in a turn the player should be able to build 5 streets,
    # so total number of streets should be 6
    for _ in range(5):
        p.hand = np.array(brd.structure.real_estate_cost[0])
        tournament.player_action_in_turn(brd, p, rejected_trades_for_this_round=dict([]))
    assert p._actions_in_round == 5
    assert sum(p.streets) == 6
    # if cannot build, proposal should be to trade, these players only like a hand for street, so they only like one
    # trade. If that is blocked for players they will trade with the bank
    p.hand = np.array([5,0,0,0,0,0])
    best_action = p.find_best_action(rejected_trades_for_this_round=set({(0,5)}))
    assert best_action[0] == 'trade_bank'
    best_action = p.find_best_action(rejected_trades_for_this_round=set({(0,1),(0,2),(0,3),(0,4),(0,5)}))
    assert best_action[0] == 'trade_bank'

def test_game_ranking():
    tournament = Tournament()
    assert all(np.array(tournament.game_ranking([4,3,2,1]))==np.array([1,2,3,4]))
    assert all(np.array(tournament.game_ranking([4,3,2,0]))==np.array([1,2,3,4]))
    assert all(np.array(tournament.game_ranking([1,2,3,4]))==np.array([4,3,2,1]))
    assert all(np.array(tournament.game_ranking([3,3,2,1]))==np.array([1,1,3,4]))
    assert all(np.array(tournament.game_ranking([3,3,3,3]))==np.array([1,1,1,1]))

def test_game_logging():
    board_layout = BoardLayout(tile_layout='DSWOSBWWGSGSBGWOGOB')
    structure = BoardStructure(board_layout=board_layout)
    structure.winning_score = 8

    players = [
        TestPlayer(name='Player1', structure=structure),
        TestPlayer(name='Player2', structure=structure),
        TestPlayer(name='Player3', structure=structure),
        TestPlayer(name='Player4', structure=structure)
    ]
    for player in players:
        player._players_in_this_game = players

    tournament = Tournament()
    tournament.list_of_orders = [[0,1,2,3]]
    tournament.list_of_reversed_orders = tournament._create_list_of_reversed_orders()
    tournament.score_table_for_ranking_per_game = [10, 5, 2, 0]
    tournament.no_games_in_tournament = 1
    tournament.verbose = False
    tournament.logging = True
    tournament.file_name_for_logging = "test_game_logging.txt"
    # We play one game with the test players. "Player1" scores 8 points and 10 victory points in 4 rounds,
    # while "Player2" scores 4 points and 5 victory points in 4 rounds. Player3 scores 2 points and 2 victory points.
    player_tournament_results, player_victory_points, rounds_for_this_game = tournament.tournament(structure, players)
    assert all(player_tournament_results == np.array([8,4,2,0]))
    assert all(player_victory_points == np.array([10,5,2,0]))
    assert all(rounds_for_this_game == np.array([4,4,4,4]))
    df = pd.read_csv("test_game_logging.txt")
    assert df.shape == (16,159)
    assert df.at[0,'turns_before_end'] == 3.0
    assert df.at[5,'node_0'] == 5.0



def test_log_vector_from_board():
    board_layout = BoardLayout(tile_layout='DSWOSBWWGSGSBGWOGOB')
    structure = BoardStructure(board_layout=board_layout)
    structure.winning_score = 8
    brd = Board(structure=structure) 
    tournament = Tournament()
    players = []
    for name in ['A','B','C','D']: 
        players.append(Player_Value_Function_Based())
    brd.players = players
    p = players[0]
    q = players[3]
    # build streets
    p.hand = np.array(brd.structure.real_estate_cost[0])
    brd.execute_player_action(p,('street',28))
    p.hand = np.array(brd.structure.real_estate_cost[0])
    brd.execute_player_action(p,('street',29))
    q.hand = np.array(brd.structure.real_estate_cost[0])
    brd.execute_player_action(q,('street',18))
    q.hand = np.array(brd.structure.real_estate_cost[0])
    brd.execute_player_action(q,('street',19))
    # build villages
    p.hand = np.array(brd.structure.real_estate_cost[1])
    brd.execute_player_action(p,('village',6))
    q.hand = np.array(brd.structure.real_estate_cost[1])
    brd.execute_player_action(q,('village',12))
    # build towns
    p.hand = np.array(brd.structure.real_estate_cost[2])
    brd.execute_player_action(p,('town',22))
    q.hand = np.array(brd.structure.real_estate_cost[2])
    brd.execute_player_action(q,('town',14))
    # fill hands
    players[1].hand = np.array(brd.structure.real_estate_cost[1])
    players[2].hand = np.array(brd.structure.real_estate_cost[2])
    # now edges 28 and 29 belong to player A, edges 18 and 19 to player D
    # player A has a village on node 6 and a town on node 22
    # player D has a village on node 12 and a town on node 14
    # A and D have empty hands
    # player B has hand for a village, and player C a hand for a town
    for p in brd.players:
        p.update_build_options()
    vector = brd.create_board_vector()
    assert len(vector) == 5 + 4 + brd.structure.no_of_nodes + brd.structure.no_of_edges + 4*6
    vector = vector[5:]  # skip first 5 elements which are not used in the log
    values = vector[:4]
    nodes = vector[4:4+brd.structure.no_of_nodes]
    edges = vector[4+brd.structure.no_of_nodes:4+brd.structure.no_of_nodes+brd.structure.no_of_edges]
    hands = vector[4+brd.structure.no_of_nodes+brd.structure.no_of_edges:]
    hands_expected = np.array([0]*6 + list(brd.structure.real_estate_cost[1])+list(brd.structure.real_estate_cost[2])+[0]*6)
    assert all(hands == hands_expected)
    streets_expected = np.zeros(brd.structure.no_of_edges,np.int32)
    streets_expected[18] = 4
    streets_expected[19] = 4
    streets_expected[28] = 1
    streets_expected[29] = 1
    assert all(edges == streets_expected)
    nodes_expected = np.zeros(brd.structure.no_of_nodes,np.int16)
    nodes_expected[6] = 1
    nodes_expected[22] = 5
    nodes_expected[12] = 4
    nodes_expected[14] = 8
    assert all(nodes == nodes_expected)