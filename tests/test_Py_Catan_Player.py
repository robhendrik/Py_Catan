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



def test_playerpreferences():
    pp = PlayerPreferences()
    assert pp.full_score == 500.0


def test_playerpreferences_randomize():
    pp = PlayerPreferences()
    d  = pp.asdict()
    for k,v in d.items():
        if k not in pp.excluded_from_normalization:
            d[k] = 1.0
    pp = PlayerPreferences(**d)
    pp = pp.normalized()
    assert all([v == d['streets'] for v in pp.asdict().items() if k not in pp.excluded_from_normalization])
    assert all([v != 1.0 for v in pp.asdict().items() if k not in pp.excluded_from_normalization])
    pp = pp.randomize_values_for_appreciation(bandwidth= 0.01)
    d  = pp.asdict()
    assert not all([v == d['streets'] for k,v in pp.asdict().items() if k not in pp.excluded_from_normalization])

def test_playertest_playerpreferences_merge():
    pp = PlayerPreferences()
    d  = pp.asdict()
    for k,v in d.items():
        if k not in pp.excluded_from_normalization:
            d[k] = 1.0
    pp = PlayerPreferences(**d)
    qq = PlayerPreferences()
    d  = qq.asdict()
    for k,v in d.items():
        if k not in qq.excluded_from_normalization:
            # create random input for setting, look whether the key contains letter 't'
            if 't' in k:
                d[k] = 1.0
            else:
                d[k] = 0.0
    qq = PlayerPreferences(**d)

    rr = pp.merge_values_for_appreciation(other_preference=qq)
    d  = rr.asdict()
    assert all([v == d['streets'] for k,v in rr.asdict().items() if (k not in pp.excluded_from_normalization and 't' in k)])
    assert all([v != d['streets'] for k,v in rr.asdict().items() if (k not in pp.excluded_from_normalization and 't' not in k)])

def test_player_copy():
    p = Player('A')
    p.hand = np.array([1,0,0,1,0,0])
    p.villages[10] = 1
    b = p.copy()
    b.hand[1] = 4
    b.hand[0] = 0
    p.villages[11] = 1
    assert b.name == 'New'
    assert p.hand[0] == 1 and p.hand[1] == 0
    assert b.hand[0] == 0 and b.hand[1] == 4 and b.hand[3] == 1
    assert b.villages[10] == 1 and b.villages[11] == 0

def test_player_longest_street():
    p = Player('A')
    b = BoardStructure()
    p.nodes_connected_by_edge = b.nodes_connected_by_edge
    for s in [ 0,1,2]:
        p.streets[s] = 1
    assert p.calculate_longest_street() == 3
    assert p.longest_street_for_this_player == 3
    for s in [0,1,2,3,7,8,9]:
        p.streets[s] = 1
    assert p.calculate_longest_street() == 4
    assert p.longest_street_for_this_player == 4
    for s in [0,1,2,3,7,8,9,10]:
        p.streets[s] = 1
    assert p.calculate_longest_street() == 5
    assert p.longest_street_for_this_player == 5
    p = Player('A')
    p.nodes_connected_by_edge = b.nodes_connected_by_edge
    for s in [0,6,7,12,13,29]:
        p.streets[s] = 1
    assert sum(p.streets) == 6
    assert p.calculate_longest_street() == 6
    assert p.longest_street_for_this_player == 6
    for s in [0,6,7,12,13,14,29]:
        p.streets[s] = 1
    assert p.calculate_longest_street() == 7
    assert p.longest_street_for_this_player == 7

def test_player_calculate_score():
    p = Player('A')
    b = BoardStructure()
    brd = Board(structure=b, players=[p])
    p.nodes_connected_by_edge = b.nodes_connected_by_edge
    p.longest_street_on_board = 1
    p.streets[10] =  1
    p.streets[11] = 1
    p.streets[12] = 1
    p.streets[13] = 1
    p.villages[11] = 1
    p.towns[10] = 1
    p.calculate_longest_street()
    assert  p.longest_street_for_this_player == 2
    assert p.calculate_score() == 3
    p.streets[14] = 1
    brd.build_street(p, 14)
    brd.build_street(p, 15)
    p.calculate_longest_street()
    assert  p.longest_street_for_this_player == 4
    assert  p.longest_street_on_board == 4
    assert p.owns_longest_street == True
    assert p.calculate_score() == 5

def test_longest_street():
    p = Player('A')
    q = Player('B')
    b = BoardStructure()
    brd = Board(structure=b, players=[p,q])
    for s in [0,1]:
        brd.build_street(p, s)
    assert p.calculate_longest_street() == 2
    assert p.owns_longest_street == False
    for s in [0,1,2]:
        brd.build_street(p, s)
    assert p.calculate_longest_street() == 3
    assert p.longest_street_for_this_player == 3
    brd._update_board_for_players()
    assert p.owns_longest_street == True
    for s in [0,1,2,3,7,8,9]:
        brd.build_street(p, s)
    assert p.calculate_longest_street() == 4
    assert p.longest_street_for_this_player == 4
    assert p.owns_longest_street == True
    for s in [0,1,2,3,7,8,9,10]:
        brd.build_street(p, s)
    assert p.calculate_longest_street() == 5
    assert p.longest_street_for_this_player == 5
    assert p.owns_longest_street == True
    for s in [54,55,56,57]:
        brd.build_street(q, s)
    assert q.calculate_longest_street() == 4
    assert q.longest_street_for_this_player == 4
    assert q.owns_longest_street == False
    for s in [58]:
        brd.build_street(q, s)
    assert q.calculate_longest_street() == 5
    brd._update_board_for_players()
    print(q.calculate_longest_street(),p.calculate_longest_street())
    assert q.longest_street_for_this_player == 5
    assert q.owns_longest_street == False
    assert p.owns_longest_street == False
    for s in [59]:
        brd.build_street(q, s)
    assert q.calculate_longest_street() == 6
    assert q.longest_street_for_this_player == 6
    assert q.owns_longest_street == True
    assert p.owns_longest_street == False
 
    
def test_can_build():
    p = Player('A')
    p.resource_types = 'BDGOSW'
    p.street_cost = 'BW'
    p.village_cost = 'BGSW'
    p.town_cost = 'GGOOO'
    p.development_card_cost = 'GOS'
    p.real_estate_cost = [
        np.array([p.street_cost.count(c) for c in 'BDGOSW']),
        np.array([p.village_cost.count(c) for c in 'BDGOSW']),
        np.array([p.town_cost.count(c) for c in 'BDGOSW']),
        np.array([p.development_card_cost.count(c) for c in 'BDGOSW'])
    ]
    p.villages[0] = 1
    p.hand = np.array([1,1,1,1,1,0])
    assert p.can_build_street() == False and p.can_build_town() == False and p.can_build_village() == False
    p.hand = np.array([1,1,1,1,1,1])
    assert p.can_build_street() == True and p.can_build_town() == False and p.can_build_village() == True
    p.hand = np.array([1,1,2,3,1,1])
    assert p.can_build_street() == True and p.can_build_town() == True and p.can_build_village() == True
    p.hand = np.array([0,1,2,3,1,1])
    assert p.can_build_street() == False and p.can_build_town() == True and p.can_build_village() == False






