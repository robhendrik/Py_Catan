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





def test_earning_power():
    b = Board()
    p = Player_Value_Function_Based()
    b.players[0] = p
    assert   all(p.calc_earning_power_for_player() == 0)
    assert len(p.calc_earning_power_for_player()) == b.structure.no_of_resource_types
    p.hand += b.structure.real_estate_cost[1]
    b.build_village(player=p,node=22)
    # 'BDGOSW'
    # 22 = O2,G5,W9 = O freq 1, G freq 4 and W freq 4 = [0,0,4,1,0,4]
    assert  all(p.calc_earning_power_for_player() == [0,0,4,1,0,4])
    p.hand += b.structure.real_estate_cost[1]
    b.build_village(player=p,node=3)
    p.hand += b.structure.real_estate_cost[2]
    b.build_town(player=p,node=3)
    # 3 = W5,S6. freq 4 en 5 [0,0,0,0,5,4]
    # town, so double
    # plus add previously build village
    assert  all(p.calc_earning_power_for_player() == [0,0,4,1,10,12])
    b._update_board_for_players()
    assert  all(p.earning_power == [0,0,4,1,10,12])
    p.preference.villages, p.preference.towns = 0,0
    p.preference.cards_earning_power = 1
    p.preference.resource_type_weight = [0,0,0,0,0,1]
    p.update_build_options()
    assert p.calculate_value() == 12
    assert p.calculate_value() == p.calculate_value(updated_version=True)
  

def test_player_calculate_value_villages():
    g = Board()
    pref = PlayerPreferences()
    pref = pref.normalized()
    g.players[0] =  Player_Value_Function_Based(preference=pref)
    g.build_village(player=g.players[0],node=1)
    g._update_board_for_players()
    g.players[0].update_build_options()
    assert g.players[0].calculate_value()==float(1/3)
    p = g.players[0]
    assert p.calculate_value() == p.calculate_value(updated_version=True)


def test_player_calculate_value_build_options():
    g = Board()
    pref = PlayerPreferences()
    pref.villages, pref.towns, pref.cards_in_hand, pref.cards_earning_power,pref.village_build_options = 0,0,0,0,1.0
    pref.resource_type_weight = np.array([1,0,1,1,1,1])
    pref = pref.normalized()
    g.players[0] =  Player_Value_Function_Based(preference=pref)
    g.build_street(player=g.players[0],edge=9)
    g.build_street(player=g.players[0],edge=20)
    g._update_board_for_players()
    g.players[0].update_build_options()
    assert g.players[0].calculate_value() == 3.0
    g.build_street(player=g.players[0],edge=12)
    g._update_board_for_players()
    g.players[0].update_build_options()
    assert g.players[0].calculate_value() == 5.0
    p = g.players[0]
    assert p.calculate_value() == p.calculate_value(updated_version=True)

def test_player_calculate_value_build_options_street():
    g = Board()
    pref = PlayerPreferences()
    pref.villages, pref.towns, pref.cards_in_hand, pref.cards_earning_power,pref.street_build_options = 0,0,0,0,1.0
    pref.resource_type_weight = np.array([1,0,1,1,1,1])
    pref = pref.normalized()
    g.players[0] =  Player_Value_Function_Based(preference=pref)
    g.build_street(player=g.players[0],edge=9)
    g.build_street(player=g.players[0],edge=20)
    g._update_board_for_players()
    g.players[0].update_build_options()
    assert g.players[0].calculate_value() == 5.0
    p = g.players[0]
    assert p.calculate_value() == p.calculate_value(updated_version=True)
    
def test_player_calculate_value_town():
    g = Board()
    pref = PlayerPreferences()
    pref = pref.normalized()
    g.players[0] =  Player_Value_Function_Based(preference=pref)
    g.build_town(player=g.players[0],node=1)
    g._update_board_for_players()
    g.players[0].update_build_options()
    assert g.players[0].calculate_value()==float(2/3)
    p = g.players[0]
    assert p.calculate_value() == p.calculate_value(updated_version=True)

def test_player_calculate_value_hand():
    g = Board()
    pref = PlayerPreferences()
    pref = pref.normalized()
    pref.villages, pref.towns, pref.cards_in_hand = 0,0,1
    pref.penalty_reference_for_too_many_cards = 0
    pref.resource_type_weight = np.array([1,0,0,0,0,0])
    g.players[0] =  Player_Value_Function_Based(preference=pref)
    g.players[0].hand = np.array([1,0,0,0,0,0])
    g._update_board_for_players()
    g.players[0].update_build_options()
    assert g.players[0].calculate_value()==1.0
    g.players[0].hand = np.array([0,1,0,0,0,0])
    assert g.players[0].calculate_value()==0.
    p = g.players[0]
    assert p.calculate_value() == p.calculate_value(updated_version=True)

def test_player_calculate_value_hand_with_penalty():
    g = Board()
    pref = PlayerPreferences()
    pref.villages, pref.towns, pref.cards_in_hand = 0,0,1.0
    pref.resource_type_weight = np.array([1,0,0,0,0,0])
    pref = pref.normalized()
    print(pref.asdict())
    g.players[0] =  Player_Value_Function_Based(preference=pref)
    g.players[0].hand = np.array([10,0,0,0,0,0])
    g._update_board_for_players()
    g.players[0].update_build_options()
    assert g.players[0].calculate_value() == 1.0 * 10 * (10/(10+7))
    g.players[0].hand = np.array([1,0,0,0,0,0])
    g._update_board_for_players()
    g.players[0].update_build_options()
    assert g.players[0].calculate_value()== 1.0 * 1.0 * (1.0/(1+7))
    p = g.players[0]
    assert p.calculate_value() == p.calculate_value(updated_version=True)

def test_player_preference_normalization_2():
    g = Board()
    pref = PlayerPreferences()
    pref.villages, pref.towns, pref.cards_in_hand, pref.cards_earning_power = 1,1,1,1
    pref.penalty_reference_for_too_many_cards = 0
    pref.resource_type_weight = np.array([1,0,1,1,1,1])
    pref = pref.normalized()
    g.players[0] =  Player_Value_Function_Based(preference=pref)
    g.players[0].hand = np.array([1,0,0,0,0,0])
    g._update_board_for_players()
    g.players[0].update_build_options()
    assert g.players[0].calculate_value()==0.2*0.25
    g.players[0].hand = np.array([0,1,0,0,0,0])
    assert g.players[0].calculate_value()==0.0
    g.players[0].hand = np.array([1,1,1,0,0,0])
    assert g.players[0].calculate_value()==0.4*0.25
    p = g.players[0]
    assert p.calculate_value() == p.calculate_value(updated_version=True)

def test_player_preference_normalization():
    g = Board()
    pref = PlayerPreferences()
    pref.villages, pref.towns, pref.cards_in_hand, pref.cards_earning_power = 1,1,1,1
    pref.penalty_reference_for_too_many_cards = 0
    pref.resource_type_weight = np.array([1,0,1,1,1,1])
    pref = pref.normalized()
    g.players[0] =  Player_Value_Function_Based(preference=pref)
    g.players[0].hand = np.array([1,0,0,0,0,0])
    g._update_board_for_players()
    g.players[0].update_build_options()
    assert g.players[0].calculate_value()==0.2*0.25
    g.players[0].hand = np.array([0,1,0,0,0,0])
    assert g.players[0].calculate_value()==0.0
    g.players[0].hand = np.array([1,1,1,0,0,0])
    assert g.players[0].calculate_value()==0.4*0.25

def test_earning_power_direct_options():
    g = Board()
    pref = PlayerPreferences()
    pref.villages, pref.towns, pref.cards_in_hand, pref.direct_options_earning_power = 0,0,0,1.0
    pref.penalty_reference_for_too_many_cards = 0
    pref.resource_type_weight = np.array([1,0,0,0,0,0])
    pref = pref.normalized()
    g.players[0] =  Player_Value_Function_Based(preference=pref)
    g.build_street(player=g.players[0],edge=11)
    g._update_board_for_players()
    g.players[0].update_build_options()
    assert (g.players[0].calculate_value())
    pref.resource_type_weight = np.array([0,0,1,1,0,0])
    pref = pref.normalized()
    g.players[0] =  Player_Value_Function_Based(preference=pref)
    g.build_street(player=g.players[0],edge=11)
    g._update_board_for_players()
    g.players[0].update_build_options()
    assert (g.players[0].calculate_value()==0)  # no resources accessible
    pref.resource_type_weight = np.array([0,0,0,0,1,0])
    pref = pref.normalized()
    g.players[0] =  Player_Value_Function_Based(preference=pref)
    g.build_street(player=g.players[0],edge=11)
    g._update_board_for_players()
    g.players[0].update_build_options()
    assert (g.players[0].calculate_value()==6) # can build twice on tile with 'S'
    p = g.players[0]
    assert p.calculate_value() == p.calculate_value(updated_version=True)

def test_player_build_explore():
    brd = Board()
    p = Player_Value_Function_Based()
    brd.players[0] = p

    brd.build_village(p,2)
    brd._update_board_for_players()
    brd.players[0].update_build_options()
    assert len(p.explore_building_town()) == 1
    assert p.explore_building_town()[0][1] == 2
    brd.build_village(p,4)
    brd._update_board_for_players()
    brd.players[0].update_build_options()
    assert len(p.explore_building_town()) == 2
    brd.build_street(p,2)
    brd.build_street(p,3)
    brd.build_village(p,2)
    brd._update_board_for_players()
    p.update_build_options()
    assert len(p.explore_building_street())== 5
    brd.build_street(p,9)
    brd.build_village(p,2)
    brd._update_board_for_players()
    p.update_build_options()
    assert len(p.explore_building_street())== 6

def test_player_build_options():
    g = Board()
    pref = PlayerPreferences()
    p = Player_Value_Function_Based(preference=pref) 
    g.players[0] = p
    p.longest_street_on_board = 2
    p.nodes_connected_by_edge = g.structure.nodes_connected_by_edge
    p.free_nodes_on_board = np.ones((p.structure.no_of_nodes),np.int16)
    p.free_edges_on_board = np.ones((p.structure.no_of_edges),np.int16)

    p.streets[44] = 1
    p.streets[45] = 1
 
    g._update_board_for_players()
    p.update_build_options()

    p.free_edges_on_board[44] = 0
    p.free_edges_on_board[45] = 0
    p.villages[1] = 1
    p.update_build_options()
    assert len(p.explore_building_town()) == 1
    assert p.explore_building_town()[0][1] == 1
    p.villages[5] = 1
    p.update_build_options()
    assert len(p.explore_building_town()) == 2
    assert len(p.explore_building_street())== 4
    assert len(p.explore_building_village())== 3

def test_trade_between_players():
    g = Board()

    p = Player_Value_Function_Based() 
    q = Player_Value_Function_Based() 
    g.players[0] = p
    g.players[1] = q
    p.hand = np.array([1,0,1,1,1,1])
    q.hand = np.array([0,0,0,0,0,0])
    g._update_board_for_players()
    p.update_build_options()
    q.update_build_options()
    rejected_trades = set([])
    for _ in range(2):
        options = p.explore_trading_with_other_player(rejected_trades=rejected_trades)
        best_action = options[0]
        response = g.propose_and_execute_trade(player = p, card_out_in = best_action[1])
        if response == False:
            rejected_trades.add(best_action[1])
    assert len(rejected_trades) == 2

    g = Board()
    pref=PlayerPreferences()
    pref.cards_in_hand = 1
    pref.resource_type_weight= np.array([1,0,0,0,0,0])
    p = Player_Value_Function_Based(preference=pref.normalized())
    pref=PlayerPreferences()
    pref.cards_in_hand = 1
    pref.resource_type_weight= np.array([0,0,0,0,0,1])
    q = Player_Value_Function_Based(preference=pref.normalized())
    g.players = [p,q]
    g._update_board_for_players()
    p.update_build_options()
    q.update_build_options()
    p.hand = np.array([1,0,1,1,1,1])
    q.hand = np.array([1,0,1,1,1,1])
    rejected_trades = set([])
    while True:
        options = p.explore_trading_with_other_player(rejected_trades=rejected_trades)
        if options:
            best_action = options[0]      
            answer = q.respond_positive_to_other_players_trading_request(card_out_in=(best_action[1][1],best_action[1][0]))
            response = g.propose_and_execute_trade(player = p, card_out_in = best_action[1])
            if response == False:
                rejected_trades.add(best_action[1])
            else:
                rejected_trades.add(tuple(reversed(best_action[1])))
        else:
            break
    assert p.hand[0] == 2 and p.hand[5] == 0
    assert q.hand[0] == 0 and q.hand[5] == 2

def test_trade_with_specific_player():
    g = Board()

    p = Player_Value_Function_Based() 
    q = Player_Value_Function_Based() 
    g.players[0] = p
    g.players[1] = q
    p.hand = np.array([1,0,1,1,1,1])
    q.hand = np.array([0,0,0,0,0,0])
    g._update_board_for_players()
    p.update_build_options()
    q.update_build_options()
    rejected_trades = set([])
    for _ in range(2):
        options = p.explore_trading_with_other_player(rejected_trades=rejected_trades)
        best_action = options[0]
        response = g.propose_and_execute_trade(player = p, 
                                               card_out_in = best_action[1],
                                                specified_trading_partner=1)
        if response == False:
            rejected_trades.add(best_action[1])
    assert len(rejected_trades) == 2

    g = Board()
    pref=PlayerPreferences()
    pref.cards_in_hand = 1
    pref.resource_type_weight= np.array([1,0,0,0,0,0])
    p = Player_Value_Function_Based(preference=pref.normalized())
    pref=PlayerPreferences()
    pref.cards_in_hand = 1
    pref.resource_type_weight= np.array([0,0,0,0,0,1])
    q = Player_Value_Function_Based(preference=pref.normalized())
    g.players = [p,q]
    g._update_board_for_players()
    p.update_build_options()
    q.update_build_options()
    p.hand = np.array([1,0,1,1,1,1])
    q.hand = np.array([1,0,1,1,1,1])
    rejected_trades = set([])
    while True:
        options = p.explore_trading_with_other_player(rejected_trades=rejected_trades)
        if options:
            best_action = options[0]      
            answer = q.respond_positive_to_other_players_trading_request(card_out_in=(best_action[1][1],best_action[1][0]))
            response = g.propose_and_execute_trade(player = p, card_out_in = best_action[1], specified_trading_partner=1)
            if response == False:
                rejected_trades.add(best_action[1])
            else:
                rejected_trades.add(tuple(reversed(best_action[1])))
        else:
            break
    assert p.hand[0] == 2 and p.hand[5] == 0
    assert q.hand[0] == 0 and q.hand[5] == 2

def test_trading_with_bank():
    g = Board()
    pref=PlayerPreferences()
    pref.cards_in_hand = 1
    pref.resource_type_weight= np.array([1,0,0,0,0,0])
    p = Player_Value_Function_Based(preference=pref.normalized())
    pref=PlayerPreferences()
    pref.cards_in_hand = 1
    pref.resource_type_weight= np.array([0,0,0,0,0,1])
    q = Player_Value_Function_Based(preference=pref.normalized())
    g.players = [p,q]
    g._update_board_for_players()
    g.players[0].update_build_options()
    g._update_board_for_players()
    p.update_build_options()
    q.update_build_options()
    p.hand = np.array([1,0,1,1,1,5])
    q.hand = np.array([5,0,1,1,1,1])

    for player in [p,q]:
        val = player.calculate_value()
        options = player.explore_trading_with_bank()
        best_action = options[0]
        if best_action[0] > val:
            card_out_in = best_action[1]
            player.hand[card_out_in[0]] -= 4
            player.hand[card_out_in[1]] += 1
    assert p.hand[0] == 2 and p.hand[5] == 1
    assert q.hand[0] == 1 and q.hand[5] == 2