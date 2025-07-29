import sys  
sys.path.append("./src/Py_Catan")
sys.path.append("./src")
import pytest
import numpy as np

from Py_Catan.BoardStructure import BoardStructure, BoardLayout
from Py_Catan.PlotBoard import PlotCatanBoard
from Py_Catan.Preferences import PlayerPreferences
import Py_Catan.Player_Preference_Types as ppt
from Py_Catan.Player import Player
from Py_Catan.Board import Board
from Py_Catan.PlayerValueFunctionBased import Player_Value_Function_Based
from Py_Catan.ValueFunction import ValueFunction
from Py_Catan.BoardVector import BoardVector


def test_value_build_options():
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
    # === check value function
    vf = ValueFunction(pref, g.structure)
    values = vf.value_for_board(g)
    assert np.array_equal([values[0]],[p.calculate_value()])
    value = vf.value_for_player(g.players[0])
    assert np.array_equal([value],[p.calculate_value()])
    board_vector = BoardVector(board=g)
    values = vf.value_for_board_vector(board_vector)
    assert np.array_equal([values[0]],[p.calculate_value()])
    vector = board_vector.vector
    values = vf.value_from_vector(vector)
    assert np.array_equal([values[0]],[p.calculate_value()])

def test_value_build_options_street():
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
    # === check value function
    vf = ValueFunction(pref, g.structure)
    values = vf.value_for_board(g)
    assert np.array_equal([values[0]],[p.calculate_value()])
    value = vf.value_for_player(g.players[0])
    assert np.array_equal([value],[p.calculate_value()])
    board_vector = BoardVector(board=g)
    values = vf.value_for_board_vector(board_vector)
    assert np.array_equal([values[0]],[p.calculate_value()])
    vector = board_vector.vector
    values = vf.value_from_vector(vector)
    assert np.array_equal([values[0]],[p.calculate_value()])
    
def test_value_town():
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
    # === check value function
    vf = ValueFunction(pref, g.structure)
    values = vf.value_for_board(g)
    assert np.allclose([values[0]],[p.calculate_value()])
    value = vf.value_for_player(g.players[0])
    assert np.allclose([value],[p.calculate_value()])
    board_vector = BoardVector(board=g)
    values = vf.value_for_board_vector(board_vector)
    assert np.allclose([values[0]],[p.calculate_value()])
    vector = board_vector.vector
    values = vf.value_from_vector(vector)
    assert np.allclose([values[0]],[p.calculate_value()])

def test_value_hand():
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
    # === check value function
    vf = ValueFunction(pref, g.structure)
    values = vf.value_for_board(g)
    assert np.allclose([values[0]],[p.calculate_value()])
    value = vf.value_for_player(g.players[0])
    assert np.allclose([value],[p.calculate_value()])
    board_vector = BoardVector(board=g)
    values = vf.value_for_board_vector(board_vector)
    assert np.allclose([values[0]],[p.calculate_value()])
    vector = board_vector.vector
    values = vf.value_from_vector(vector)
    assert np.allclose([values[0]],[p.calculate_value()])

def test_value_hand_with_penalty():
    g = Board()
    pref = PlayerPreferences()
    pref.villages, pref.towns, pref.cards_in_hand = 0,0,1.0
    pref.resource_type_weight = np.array([1,0,0,0,0,0])
    pref = pref.normalized()
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
    # === check value function
    vf = ValueFunction(pref, g.structure)
    values = vf.value_for_board(g)
    assert np.allclose([values[0]],[p.calculate_value()])
    value = vf.value_for_player(g.players[0])
    assert np.allclose([value],[p.calculate_value()])
    board_vector = BoardVector(board=g)
    values = vf.value_for_board_vector(board_vector)
    assert np.allclose([values[0]],[p.calculate_value()])
    vector = board_vector.vector
    values = vf.value_from_vector(vector)
    assert np.allclose([values[0]],[p.calculate_value()])

def test_value_generic():
    g = Board()
    pref = ppt.mediocre_1
    g.players = [
        Player_Value_Function_Based(preference=pref),
        Player_Value_Function_Based(preference=pref),
        Player_Value_Function_Based(preference=pref),
        Player_Value_Function_Based(preference=pref)]

    g.players[0].hand = np.array([5,5,5,5,5,5])
    g.players[1].hand = np.array([5,5,5,5,5,5])
    g.players[2].hand = np.array([5,5,5,5,5,5])
    g.players[3].hand = np.array([5,5,5,5,5,5])
    g.execute_player_action(player = g.players[0], best_action = ('street', 9))
    g.execute_player_action(player = g.players[1], best_action = ('street', 20))
    g.execute_player_action(player = g.players[2], best_action = ('street', 12))
    g.execute_player_action(player = g.players[3], best_action = ('street', 15))
    g.execute_player_action(player = g.players[0], best_action = ('village', 1))
    g.execute_player_action(player = g.players[1], best_action = ('village', 20))
    g.execute_player_action(player = g.players[2], best_action = ('village', 30))
    g.execute_player_action(player = g.players[3], best_action = ('village', 40))
    g.execute_player_action(player = g.players[0], best_action = ('town', 1))
    g.players[0].hand = np.array([5,0,1,2,3,5])
    g.players[1].hand = np.array([1,0,2,3,4,5])
    g.players[2].hand = np.array([1,0,1,1,1,1])
    g.players[3].hand = np.array([0,0,0,0,0,0])
    g._update_board_for_players()
    for player in g.players:
        player.update_build_options()
    values_from_players = [player.calculate_value() for player in g.players]
    # === check value function
    vf = ValueFunction(preference = pref, structure = g.structure)
    values = vf.value_for_board(g)
    assert np.allclose(values,values_from_players)
    values = [vf.value_for_player(p) for p in g.players]
    assert np.allclose(values,values_from_players)
    board_vector = BoardVector(board=g)
    values = vf.value_for_board_vector(board_vector)
    assert np.allclose(values,values_from_players)
    vector = board_vector.vector
    values = vf.value_from_vector(vector)
    assert np.allclose(values,values_from_players)

