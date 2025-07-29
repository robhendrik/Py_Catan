import sys  
sys.path.append("./src/Py_Catan")
sys.path.append("./src")
import pytest
import numpy as np

from Py_Catan.Board import Board
from Py_Catan.BoardStructure import BoardStructure, BoardLayout
from Py_Catan.Player import Player
from Py_Catan.Preferences import PlayerPreferences

def test_board_initialization():
    # Test with default structure and players
    board = Board()
    assert board.structure is not None
    assert board.structure.tile_layout == 'DSWSWSWWGSOBGBGOBOG'
    assert len(board.players) == 4
    assert all(isinstance(player, Player) for player in board.players)

    # Test with custom structure and players
    structure = BoardStructure()
    players = [Player(name=f'Player {i}', preference=PlayerPreferences(), structure=structure) for i in range(3)]
    board = Board(structure=structure, players=players)
    assert board.structure == structure
    assert len(board.players) == 3

def test_dice_impact():
    board_layout = BoardLayout()
    board_layout.tile_layout = 'DSWSWSWWGSOBGBGOBOG'
    board_layout.values = [0,9,4,5,6,3,11,5,2,6,3,8,10,9,12,11,4,8,10]
    board = BoardStructure(board_layout=board_layout)
    for node in range(6):   
        assert board.dice_impact_per_node_dnt[0][node][1] == 1 and board.dice_impact_per_node_dnt[0][node][0] == 0

    tile_number = 16
    for node in board.neighbour_nodes_for_tiles[tile_number]:   
        resource_1 = board.resource_types.index('B')
        resource_2 = board.resource_types.index('W')
        dice_result = board.dice_results.index(4)
        assert board.dice_impact_per_node_dnt[dice_result][node][resource_1] == 1 and board.dice_impact_per_node_dnt[dice_result][node][resource_2] == 0

    tile_number = 5
    for node in board.neighbour_nodes_for_tiles[tile_number]:   
        resource_1 = board.resource_types.index('S')
        resource_2 = board.resource_types.index('W')
        dice_result = board.dice_results.index(3)
        assert board.dice_impact_per_node_dnt[dice_result][node][resource_1] == 1 and board.dice_impact_per_node_dnt[dice_result][node][resource_2] == 0

    assert all([board.node_earning_power[0][i] == [0, 0, 0, 0, 4, 2][i] for i in range(6)])
    assert all([board.node_earning_power[6][i] == [0, 0, 3, 0, 4, 4][i] for i in range(6)])
    assert all([board.node_earning_power[23][i] == [0, 0, 3, 0, 4, 2][i] for i in range(6)])

def test_generic_board_numbers():
    board_layout = BoardLayout()
    board_layout.tile_layout = 'DSWSWSWWGSOBGBGOBOG'
    board_layout.values = [0,9,4,5,6,3,11,5,2,6,3,8,10,9,12,11,4,8,10]
    board = BoardStructure(board_layout=board_layout)
    assert len(board._tile_coordinates) == 19
    assert len(board._node_coordinates) == 54
    assert len(board._edge_coordinates) == 72

def test_generic_board_neighbour_nodes_for_tiles():
    board_layout = BoardLayout()
    board_layout.tile_layout = 'DSWSWSWWGSOBGBGOBOG'
    board_layout.values = [0,9,4,5,6,3,11,5,2,6,3,8,10,9,12,11,4,8,10]
    board = BoardStructure(board_layout=board_layout)
    cases = {
        (0,0) : [(0,0),(0,1),(0,2),(0,3),(0,4),(0,5)],
        (2,0) : [(1,0),(1,1),(2,29),(2,0),(2,1),(2,2)],
        (0,3) : [(0,3),(0,4),(1,8),(1,9),(1,10),(1,11)]
        }

    for k,vs in cases.items():
        tile = board.polar_to_tile((k[0],k[1]))
        assert len(vs) == len(board.neighbour_nodes_for_tiles[tile])
        for v in vs:
            node = board.polar_to_node((v[0],v[1]))
            assert node in board.neighbour_nodes_for_tiles[tile]

def test_polar_to_node():
    b = BoardStructure()
    assert 0 == b.polar_to_node((0,0))
    assert 1 == b.polar_to_node((0,1))
    assert 25 == b.polar_to_node((2,1))
    assert 6 == b.polar_to_node((1,0))

def test_polar_to_tile():
    b = BoardStructure()
    assert 1 == b.polar_to_tile((1,0))
    assert 2 == b.polar_to_tile((1,1))
    assert 7 == b.polar_to_tile((2,0))
    assert 18 == b.polar_to_tile((2,11))

def test_polar_to_edge():
    b = BoardStructure()
    assert 0 == b.polar_to_edge((0,0))
    assert 30 == b.polar_to_edge((2,0))
    assert 42 == b.polar_to_edge((2,12))

def test_generate_boards():
    b = BoardStructure()
    l = b.generate_list_of_all_possible_boards()
    boards = ['DSWSWSWWGSOBGBGOBOG','DWGWBGOOBWBSGSWOSGS','DSWSOGOGOGBGBSWSWBW','DSGSBWOBWOBWGSOGSWG']
    for board in boards:
        assert board in l
        assert len(board) == len(b._tile_coordinates)
        for node in range(len(b._node_coordinates)):
            for d in range(2,13):
                if d != 7:
                    dice_result = b.dice_results.index(d)
                    assert max(b.dice_impact_per_node_dnt[dice_result,node]) <= 1

def test_dice_effect_for_node():
    board = 'DSWSWSWWGSOBGBGOBOG'
    board_layout = BoardLayout(tile_layout=board)
    b = BoardStructure(board_layout=board_layout)
    d = 10 # expext W for 2,0
    resource_type = 'W'
    dice_result = b.dice_results.index(d)
    resource_index = b.resource_types.index(resource_type)
    old_notation = (2,0)
    node_index = b.polar_to_node(old_notation)
    assert b.dice_impact_per_node_dnt[dice_result][node_index][resource_index] == 1

    d =11 # expect S for 1,0
    old_notation = (1,0)
    resource_type = 'S'
    dice_result = b.dice_results.index(d)
    resource_index = b.resource_types.index(resource_type)
    node_index = b.polar_to_node(old_notation)
    assert b.dice_impact_per_node_dnt[dice_result][node_index][resource_index] == 1

    d =9 # expect W for 0,5
    old_notation = (0,5)
    resource_type = 'W'
    dice_result = b.dice_results.index(d)
    resource_index = b.resource_types.index(resource_type)
    node_index = b.polar_to_node(old_notation)
    assert b.dice_impact_per_node_dnt[dice_result][node_index][resource_index] == 1
 