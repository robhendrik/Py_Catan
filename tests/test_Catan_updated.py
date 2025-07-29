import sys  
sys.path.append("./src/Py_Catan")
sys.path.append("./src")
import pytest
import numpy as np

from Py_Catan.Board import Board
from Py_Catan.BoardStructure import BoardStructure, BoardLayout
from Py_Catan.Player import Player
from Py_Catan.Preferences import PlayerPreferences
from Py_Catan.PlotBoard import PlotCatanBoard

