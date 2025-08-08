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
from Py_Catan.ReferenceBoards import create_reference_board_1
import Py_Catan.Player_Preference_Types as pppt

def test_generic_board_numbers():
    b = Board()
    assert b.structure.no_of_nodes == 54
    assert len(b.structure.tile_layout) == 19
    assert len(b.structure.values) == 19
    assert b.structure.no_of_edges == 72

def test_free_edges():
    b = Board()
    assert all([s == 1 for s in b.free_edges()])
    b.build_street(player=b.players[1],edge=10 )
    assert all([s == 0 if i == 10 else 1 for i,s in enumerate(b.free_edges())])

def test_free_nodes():
    b = Board()
    assert all([s == 0 for s in b.occupied_nodes])
    assert all([s == 1 for s in b.free_nodes()])
    b.build_village(player=b.players[1],node=10 )
    assert all([s == 1 if i == 10 else s == 0 for i,s in enumerate(b.occupied_nodes)])
    neighbours = [9,11,31]
    assert all([s == 0 if i in (neighbours + [10]) else s == 1 for i,s in enumerate(b.free_nodes())])

def test_connected_edges():
    b = Board()
    strts = np.zeros(b.structure.no_of_edges,np.int16)
    strts[0] = 1
    connections = strts @ b.structure.edge_edge_matrix
    for c in [5,6,7,1]:
        assert connections[c] == 1
    for c in [0,29,13,3]:
        assert connections[c] == 0

def test_board_build_town():
    b = Board()
    b.players[0].hand = np.array([10,10,10,10,10,10])
    b.build_village(player=b.players[0],node=10)
    b.build_town(player=b.players[0],node=10)
    assert b.occupied_nodes[10] == 1
    assert b.players[0].towns[10] == 1 and b.players[0].villages[10] == 0
    occupied = [9,11,31]
    free = b.free_nodes()
    for o in occupied:
        assert b.occupied_nodes[o] == 0 and free[o] == 0
    #'BDGOSW'
    assert b.players[0].hand[0] == 9
    assert b.players[0].hand[1] == 10
    assert b.players[0].hand[2] == 7
    assert b.players[0].hand[3] == 7
    assert b.players[0].hand[4] == 9
    assert b.players[0].hand[5] == 9

def test_board_build_village():
    board = ['DSWSWSWWGSOBGBGOBOG','DWGWBGOOBWBSGSWOSGS','DSWSOGOGOGBGBSWSWBW','DSGSBWOBWOBWGSOGSWG']
    
    b = Board(structure=BoardStructure(BoardLayout(tile_layout=board[1])))
    b.players[2].hand = np.array([10,10,10,10,10,10])
    b.build_village(player=b.players[2],node=10)
    assert b.occupied_nodes[10] == 1
    occupied = [9,11,31]
    free = b.free_nodes()
    for o in occupied:
        assert b.occupied_nodes[o] == 0 and free[o] == 0
    #'BDGOSW'
    assert b.players[2].hand[0] == 9
    assert b.players[2].hand[1] == 10
    assert b.players[2].hand[2] == 9
    assert b.players[2].hand[3] == 10
    assert b.players[2].hand[4] == 9
    assert b.players[2].hand[5] == 9

def test_board_build_village_and_streets():
    board = ['DSWSWSWWGSOBGBGOBOG','DWGWBGOOBWBSGSWOSGS','DSWSOGOGOGBGBSWSWBW','DSGSBWOBWOBWGSOGSWG']
    b = Board(structure=BoardStructure(BoardLayout(tile_layout=board[2])))
    b.players[1].hand = np.array([10,10,10,10,10,10])
    nodes = [18,19,20]
    edges = [24,25]
    b.build_village(player=b.players[1],node=nodes[0])
    b.build_street(player=b.players[1],edge=edges[0] )
    b.build_street(player=b.players[1],edge=edges[1] )
    assert b.players[1].longest_street_for_this_player == 2
    assert b.occupied_nodes[18] == 1
    assert b.occupied_edges[24] == 1
    assert b.occupied_edges[25] == 1
    free = b.free_nodes()
    assert free[20] == 1
    assert free[19] == 0
    b.players[1].free_nodes_on_board = b.free_nodes()
    b.players[1].free_edges_on_board = b.free_edges()   
    assert b.players[1].free_edges_on_board[25] == 0
    assert b.players[1].free_edges_on_board[26] == 1

    b.players[1].update_build_options()
    build_options_villages = b.players[1].build_options['village_options']
    assert build_options_villages[20] == 1 and build_options_villages[19] == 0
    assert build_options_villages[23] == 0 and build_options_villages[22] == 0

    build_options_street =  b.players[1].build_options['street_options']
    assert build_options_street[24] == 0
    assert build_options_street[25] == 0
    assert build_options_street[11] == 1
    assert build_options_street[26] == 1
    #'BDGOSW'
    assert b.players[1].hand[0] == 7
    assert b.players[1].hand[1] == 10
    assert b.players[1].hand[2] == 9
    assert b.players[1].hand[3] == 10
    assert b.players[1].hand[4] == 9
    assert b.players[1].hand[5] == 7
    assert b.players[1].longest_street_for_this_player == 2
    for p in b.players:
        assert p.longest_street_on_board == 3



def test_build_options():
    b = Board()
    b.build_village(player = b.players[0],node=40)
    b.build_town(player = b.players[0],node=40)
    b.build_village(player = b.players[0],node=38)
    b.build_street(player = b.players[0],edge = 55)
    b.build_street(player = b.players[0],edge = 56)
    b.build_street(player = b.players[0],edge = 57)
    for p in b.players:
        assert p.longest_street_on_board == 3
    p = b.players[0]
    b._update_board_for_players()
    p.update_build_options()
    assert p.towns[40] == 1
    assert p.villages[38] == 1
    s = p.streets
    assert np.count_nonzero(s == 1) == 3
    for e in [55,56,57]:
        assert s[e] == 1
    options = p.build_options
    print(options['village_options'])
    assert np.count_nonzero(options['street_options'] == 1)==3
    assert np.count_nonzero(options['secondary_village_options'] == 1)== 1
    for e in  [36]:
        assert options['secondary_village_options'][e] == 1

def test_throw_dice():
    g = Board()
    dv = g.throw_dice(enforce=5)    
    assert sum(g.players[0].hand) == 0
    
    brd = Board()
    brd.players[0].hand = np.array(brd.structure.real_estate_cost[1])
    brd.build_village(brd.players[0],8)
    dv = brd.throw_dice(enforce=11)    
    assert  brd.players[0].hand[brd.structure.resource_types.index('S')] == 1

    brd = Board()
    brd.players[0].hand =2*np.array(brd.structure.real_estate_cost[1])+np.array(brd.structure.real_estate_cost[2])
    brd.build_village(brd.players[0],10)
    brd.build_town(brd.players[0],10)
    brd.build_village(brd.players[0],8)
    dv = brd.throw_dice(enforce=11)    
    dv = brd.throw_dice(enforce=3)    
    
    assert  brd.players[0].hand[brd.structure.resource_types.index('O')] == 2
    assert  brd.players[0].hand[brd.structure.resource_types.index('S')] == 1
    assert  brd.players[0].hand[brd.structure.resource_types.index('W')] == 3

def test_throw_dice_value_7():
    brd = Board()
    brd.players[0].hand = np.array([1,0,1,1,1,1])
    brd.throw_dice(enforce=7)
    assert sum(brd.players[0].hand ) == 5
    brd.players[0].hand = np.array([2,0,2,2,3,3])
    brd.throw_dice(enforce=7)
    assert sum(brd.players[0].hand ) == 6
    brd.players[0].hand = np.array([2,0,2,3,3,3])
    brd.throw_dice(enforce=7)
    assert sum(brd.players[0].hand ) == 7
   
def test_create_and_retrieve_board_status():
    brd = Board()
    brd.players[0].hand =2*np.array(brd.structure.real_estate_cost[1])+np.array(brd.structure.real_estate_cost[2])
    brd.build_village(brd.players[0],10)
    brd.build_town(brd.players[0],10)
    brd.build_village(brd.players[0],8)
    brd.players[1].hand =2*np.array(brd.structure.real_estate_cost[1])+np.array(brd.structure.real_estate_cost[2])
    brd.players[3].name ="checkcheck"
    brd.players[3].preference.hand_for_street = 10

    status = brd.create_board_status()
    brd2 = Board(status=status)
    assert brd2.players[3].name == "checkcheck"
    assert all(brd2.players[1].hand == 2*np.array(brd.structure.real_estate_cost[1])+np.array(brd.structure.real_estate_cost[2]))
    assert sum(brd2.players[0].villages) == 1
    assert sum(brd2.players[0].towns) == 1
    assert brd2.players[0].preference.hand_for_street == 0
    assert brd2.players[3].preference.hand_for_street == 10


def test_recreate_board():
    old_board = create_reference_board_1()
    old_board.inform_players_of_the_board_and_position()
    old_values = np.array([p.value_function.value_for_player(p) for p in old_board.players])
    #
    new_board = old_board.recreate()
    #
    new_player = Player_Value_Function_Based(name = 'optimized_1+',structure=old_board.structure,preference=pppt.optimized_1_with_0_for_full_score)
    new_player.copy_position_from_other_player(old_board.players[0])
    new_board.players[0] = new_player
  
    new_player = Player_Value_Function_Based(name = 'optimized_2+',structure=old_board.structure,preference=pppt.optimized_1)
    new_player.copy_position_from_other_player(old_board.players[1])
    new_board.players[1] = new_player

    new_player = Player_Value_Function_Based(name = 'optimized_3+',structure=old_board.structure,preference=pppt.optimized_2)
    new_player.copy_position_from_other_player(old_board.players[2])
    new_board.players[2] = new_player
  
    new_player = Player_Value_Function_Based(name = 'optimized_4+',structure=old_board.structure,preference=pppt.optimized_1)
    new_player.copy_position_from_other_player(old_board.players[3])
    new_board.players[3] = new_player
 
    new_board.inform_players_of_the_board_and_position()
    new_board.sync_status_between_board_and_players()
    assert np.all(old_board.occupied_edges == new_board.occupied_edges), "Occupied edges do not match"
    assert np.all(old_board.occupied_nodes == new_board.occupied_nodes), "Occupied nodes do not match"

    for old_player, new_player in zip(old_board.players, new_board.players):
        assert type(old_player) == type(new_player), "Player types do not match"
        assert old_player.preference == new_player.preference, "Preferences do not match"
        assert old_player.owns_longest_street == new_player.owns_longest_street, "Ownership of longest street does not match"
        assert old_player.longest_street_for_this_player == new_player.longest_street_for_this_player, "Longest street for player does not match"
        assert np.array_equal(old_player.hand, new_player.hand), "Hands do not match"
        assert np.array_equal(old_player.streets, new_player.streets), "Streets do not match"
        assert np.array_equal(old_player.towns, new_player.towns), "Towns do not match"
        assert np.array_equal(old_player.villages, new_player.villages), "Villages do not match"
        assert old_player._player_position == new_player._player_position, "Player positions do not match"     
        assert np.array_equal(old_player.build_options['street_options'],new_player.build_options['street_options']), "Build options do not match"
        assert np.array_equal(old_player.build_options['village_options'],new_player.build_options['village_options']), "Build options do not match"
        assert np.array_equal(old_player.build_options['secondary_village_options'],new_player.build_options['secondary_village_options']), "Build options do not match"
        assert np.array_equal(old_player.free_edges_on_board, new_player.free_edges_on_board), "Free edges on board do not match"
        assert np.array_equal(old_player.free_nodes_on_board, new_player.free_nodes_on_board), "Free nodes on board do not match"

    new_values = np.array([p.value_function.value_for_player(p) for p in old_board.players])
    assert np.array_equal(old_values, new_values), "Values do not match after recreation"