import sys  
sys.path.append("./src/Py_Catan")
sys.path.append("./src")
import pytest
import numpy as np

#from Py_Catan.GenBoard import GenBoard
from Py_Catan.BoardStructure import BoardStructure
from Py_Catan.Board import Board
from Py_Catan.Preferences import PlayerPreferences
from Py_Catan.Player import Player
from Py_Catan.PlotBoard import PlotCatanBoard

from unittest.mock import Mock
import matplotlib.pyplot as plt
from unittest.mock import patch
import matplotlib.testing.compare as plt_test

generate_reference_images =False

def test_plot_board_indicators():
    img1 = "./tests/test_drawings/testdrawing_plot_board_indicators.png"
    img2 = img1.replace("testdrawing","reference")
    if not generate_reference_images:
        def save_drawing():
            plt.savefig(img1, bbox_inches='tight')
            return
    else:
        def save_drawing():
            plt.savefig(img2, bbox_inches='tight')
            return
    with patch("Py_Catan.PlotBoard.plt.show", wraps=save_drawing) as mock_bar:
        b = Board()
        b.build_village(player = b.players[0],node=40)
        b.build_town(player = b.players[0],node=40)
        b.build_village(player = b.players[0],node=38)
        b.build_street(player = b.players[0],edge = 55)
        b.build_street(player = b.players[0],edge = 56)
        b.build_street(player = b.players[0],edge = 57)
        b.build_village(player = b.players[1],node=21)
        b.build_town(player = b.players[1],node = 21)
        b.build_street(player = b.players[2],edge = 0)
        b.build_street(player = b.players[3],edge = 11)
        b.players[0].hand = np.array([1,1,1,1,1,1],np.int16)  
        b.players[1].hand = np.array([1,0,1,0,1,0],np.int16)  
        b.players[2].hand = np.array([1,2,3,4,5,6],np.int16)  
        b.players[3].hand = np.array([1,0,1,3,1,1],np.int16) 
        draw = PlotCatanBoard(board=b)
        draw._plot_board_indicators()
        #draw.plot_board()
        #draw.plot_board_positions()
        assert plt_test.compare_images(img1, img2, 0.001) is None

def test_plot_board():
    img1 = "./tests/test_drawings/testdrawing_plot_board.png"
    img2 = img1.replace("testdrawing","reference")
    if not generate_reference_images:
        def save_drawing():
            plt.savefig(img1, bbox_inches='tight')
            return
    else:
        def save_drawing():
            plt.savefig(img2, bbox_inches='tight')
            return
    with patch("Py_Catan.PlotBoard.plt.show", wraps=save_drawing) as mock_bar:
        b = Board()
        b.build_village(player = b.players[0],node=40)
        b.build_town(player = b.players[0],node=40)
        b.build_village(player = b.players[0],node=38)
        b.build_street(player = b.players[0],edge = 55)
        b.build_street(player = b.players[0],edge = 56)
        b.build_street(player = b.players[0],edge = 57)
        b.build_village(player = b.players[1],node=21)
        b.build_town(player = b.players[1],node = 21)
        b.build_street(player = b.players[2],edge = 0)
        b.build_street(player = b.players[3],edge = 11)
        b.players[0].hand = np.array([1,1,1,1,1,1],np.int16)  
        b.players[1].hand = np.array([1,0,1,0,1,0],np.int16)  
        b.players[2].hand = np.array([1,2,3,4,5,6],np.int16)  
        b.players[3].hand = np.array([1,0,1,3,1,1],np.int16) 
        draw = PlotCatanBoard(board=b)
        #draw._plot_board_indicators()
        draw.plot_board(number=11)
        #draw.plot_board_positions()
        assert plt_test.compare_images(img1, img2, 0.001) is None

def test_plot_board_positions():
    img1 = "./tests/test_drawings/testdrawing_plot_board_positions.png"
    img2 = img1.replace("testdrawing","reference")
    if not generate_reference_images:
        def save_drawing():
            plt.savefig(img1, bbox_inches='tight')
            return
    else:
        def save_drawing():
            plt.savefig(img2, bbox_inches='tight')
            return
    with patch("Py_Catan.PlotBoard.plt.show", wraps=save_drawing) as mock_bar:
        b = Board()
        b.build_village(player = b.players[0],node=40)
        b.build_town(player = b.players[0],node=40)
        b.build_village(player = b.players[0],node=38)
        b.build_street(player = b.players[0],edge = 55)
        b.build_street(player = b.players[0],edge = 56)
        b.build_street(player = b.players[0],edge = 57)
        b.build_village(player = b.players[1],node=21)
        b.build_town(player = b.players[1],node = 21)
        b.build_street(player = b.players[2],edge = 0)
        b.build_street(player = b.players[3],edge = 11)
        b.players[0].hand = np.array([1,1,1,1,1,1],np.int16)  
        b.players[1].hand = np.array([1,0,1,0,1,0],np.int16)  
        b.players[2].hand = np.array([1,2,3,4,5,6],np.int16)  
        b.players[3].hand = np.array([1,0,1,3,1,1],np.int16) 
        draw = PlotCatanBoard(board=b)
        #draw._plot_board_indicators()
        #draw.plot_board()
        draw.plot_board_positions()
        assert plt_test.compare_images(img1, img2, 0.001) is None