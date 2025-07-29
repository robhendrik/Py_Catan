import sys  
sys.path.append("../src")
import numpy as np
import matplotlib.pyplot as plt

from Py_Catan.Board import Board
from Py_Catan.BoardStructure import BoardStructure, BoardLayout
from Py_Catan.Player import Player
from Py_Catan.Preferences import PlayerPreferences
from Py_Catan.PlayerValueFunctionBased import Player_Value_Function_Based
from Py_Catan.PlayerRandom import Player_Random
from Py_Catan.PlayerPassive import Player_Passive
from Py_Catan.PlotBoard import PlotCatanBoard
from Py_Catan.Tournament import Tournament
import Py_Catan.Player_Preference_Types as pppt

class BoardVector:
    def __init__(self, board: Board = Board(), include_values: bool = True):
        self.board = board
        self.structure = board.structure
        self.players = board.players
        self.include_values = include_values
        self.indices = self.get_indices()
        self.vector = self.create_vector_from_board()
        

    def header(self) -> list[str]:
        '''
        Create a header for the vector representation of the board.
        The header contains the names of the attributes in the vector.

        Returns:
            list[str]: A list of strings representing the header of the vector.
        '''
        header = self.structure.header
        return header
        # header = [
        #     'turns_before_end',
        #     'rank_A',
        #     'rank_B',
        #     'rank_C',
        #     'rank_D',
        #     'value_A',
        #     'value_B',
        #     'value_C',
        #     'value_D'
        # ]
        # header += ['node_'+str(n) for n in range(self.structure.no_of_nodes)]
        # header += ['egde_'+str(n) for n in range(self.structure.no_of_edges)]
        # header += ['hand_A_'+str(n) for n in range(len(self.players[0].hand))]
        # header += ['hand_B_'+str(n) for n in range(len(self.players[1].hand))]
        # header += ['hand_C_'+str(n) for n in range(len(self.players[2].hand))]
        # header += ['hand_D_'+str(n) for n in range(len(self.players[3].hand))]
        # return header           
    
    def get_vector_size(self) -> int:
        '''
        Get the size of the vector representation of the board.
        The size is the number of attributes in the vector.

        Returns:
            int: The size of the vector.
        '''
        return len(self.structure.header)
    
    def get_indices(self) -> dict:
        '''
        Create a dictionary with the indices of the attributes in the vector.
        The indices are used to access the attributes in the vector.

        Returns:
            dict: A dictionary with the indices of the attributes in the vector.
        '''
        return self.structure.vector_indices

        # indices = {
        #     'turns_before_end': 0,
        #     'rank_A': 1,
        #     'rank_B': 2,
        #     'rank_C': 3,
        #     'rank_D': 4,
        #     'value_A': 5,
        #     'value_B': 6,
        #     'value_C': 7,
        #     'value_D': 8
        # }
        # for i in range(self.structure.no_of_nodes):
        #     indices['node_'+str(i)] = i + 9
        # for i in range(self.structure.no_of_edges):
        #     indices['edge_'+str(i)] = i + 9 + self.structure.no_of_nodes
        # for i in range(len(self.players[0].hand)):
        #     indices['hand_A_'+str(i)] = i + 9 + self.structure.no_of_nodes + self.structure.no_of_edges
        # for i in range(len(self.players[1].hand)):
        #     indices['hand_B_'+str(i)] = i + 9 + self.structure.no_of_nodes + self.structure.no_of_edges + len(self.players[0].hand)
        # for i in range(len(self.players[2].hand)):
        #     indices['hand_C_'+str(i)] = i + 9 + self.structure.no_of_nodes + self.structure.no_of_edges + len(self.players[0].hand) + len(self.players[1].hand)
        # for i in range(len(self.players[3].hand)):
        #     indices['hand_D_'+str(i)] = i + 9 + self.structure.no_of_nodes + self.structure.no_of_edges + len(self.players[0].hand) + len(self.players[1].hand) + len(self.players[2].hand)
        # indices['turns'] = [indices['turns_before_end']]
        # indices['ranks'] = [indices['rank_A'], indices['rank_B'], indices['rank_C'], indices['rank_D']]
        # indices['values'] = [indices['value_A'], indices['value_B'], indices['value_C'], indices['value_D']]
        # indices['nodes'] = [indices['node_'+str(i)] for i in range(self.structure.no_of_nodes)]
        # indices['edges'] = [indices['edge_'+str(i)] for i in range(self.structure.no_of_edges)]
        # indices['hands'] = [indices[f'hand_{chr(65+i)}_{j}'] for i in range(len(self.players)) for j in range(len(self.players[0].hand))]
        # indices['hand_for_player'] = [[indices[f'hand_{chr(65+i)}_{j}'] for j in range(self.structure.no_of_resource_types)] for i in range(len(self.players)) ]
        # return indices
    
    def create_vector_from_board(self) -> np.array:
        '''
        Create a vector representation of the board.
        The vector contains the occupied nodes and edges, and the players' hands.
        - Game related entries are set to 0 (e.g. turns before end, ranks).
        - Player values are set to 0 if self.include_values is False.
        - Nodes are set to 0 if they are not occupied by a player.
            * If a node is occupied by a player, it is set to the player's index + 1 for villages and index + 5 for towns.
        - Edges are set to 0 if they are not occupied by a player.
            * If an edge is occupied by a player, it is set to the player's index + 1.
        - Players' hands are set to the number of resources they have.

        Returns:
            np.array: A numpy array representing the board.
        '''
        return self.board.create_board_vector(include_values=self.include_values)
        # precursor = np.zeros(self.indices['values'][0])
        # if self.include_values:
        #     values = [p.calculate_value() for p in self.players]
        # else:
        #     values = [0, 0, 0, 0]
        # nodes = np.zeros(self.structure.no_of_nodes)
        # for index,p in enumerate(self.players):
        #     nodes += (index+1) * p.villages
        #     nodes += (index + 5) * p.towns
        # edges = np.zeros(self.structure.no_of_edges)
        # for index,p in enumerate(self.players):
        #     edges += (index+1) * p.streets
        # hands = np.concatenate([p.hand for p in self.players])
        
        # return np.concatenate([precursor,values,nodes,edges,hands])
    
    def create_board_from_vector(self) -> Board:
        '''
        Create a board with value based players from a vector representation.
        The vector contains the occupied nodes and edges, and the players' hands.

        Returns:
            Board: A Board object with value based players.
        '''
        precursor = self.vector[:self.indices['values'][0]]
        values = self.vector[self.indices['values']]
        nodes = self.vector[self.indices['nodes']]
        edges = self.vector[self.indices['edges']]
        hands = self.vector[self.indices['hands']]
        
        occupied_nodes = np.zeros(self.structure.no_of_nodes)
        occupied_edges = np.zeros(self.structure.no_of_edges)
        players = []
        for i in range(4):
            player = Player_Value_Function_Based(name=f'Player_{i+1}', structure=self.structure)
            player.villages = np.where(nodes == i+1, 1, 0)
            player.towns = np.where(nodes == i+5, 1, 0)
            player.streets = np.where(edges == i+1, 1, 0)
            player.hand = self.vector[self.indices['hand_for_player'][i]]
            occupied_edges = np.logical_or(occupied_edges, player.streets)
            occupied_nodes = np.logical_or(occupied_nodes, player.villages + player.towns)
            players.append(player)
        board = Board(structure=self.structure, players=players)
        board.occupied_nodes = occupied_nodes
        board.occupied_edges = occupied_edges
        board._update_board_for_players()
        
        return board

    def build_street(self, player_position: int, edge_index: int) -> np.array:
        '''
        Build a street for the player on the edge with the given index.
        Removes the cost of the street from the player's hand and updates the board vector.

        Function returns a new vector as np.array with the updated resources and the street built.

        Returns:
            np.array: A new vector with the updated resources and the street built.
        '''
        new_vector = self.vector.copy()
        cost = np.array(self.structure.real_estate_cost[0])
        indices_for_hand = self.indices['hand_for_player'][player_position]
        index_for_edge = self.indices['edges'][edge_index]
        new_vector[index_for_edge] = player_position+1
        new_vector[indices_for_hand] -= cost
        return new_vector
    
    def build_village(self, player_position: int, node_index: int) -> np.array:
        '''
        Build a village for the player on the node with the given index.
        Removes the cost of the village from the player's hand and updates the board vector.

        Function returns a new vector as np.array with the updated resources and the village built.

        Returns:
            np.array: A new vector with the updated resources and the village built.
        '''
        new_vector = self.vector.copy()
        cost = np.array(self.structure.real_estate_cost[1])
        indices_for_hand = self.indices['hand_for_player'][player_position]
        index_for_node = self.indices['nodes'][node_index]
        new_vector[index_for_node] = player_position+1
        new_vector[indices_for_hand] -= cost
        return new_vector
    
    def build_town(self, player_position: int, node_index: int) -> np.array:
        '''
        Build a town for the player on the node with the given index.
        Removes the cost of the town from the player's hand and updates the board vector.

        Function returns a new vector as np.array with the updated resources and the town built.

        Returns:
            np.array: A new vector with the updated resources and the town built.
        '''
        new_vector = self.vector.copy()
        cost = np.array(self.structure.real_estate_cost[2])
        indices_for_hand = self.indices['hand_for_player'][player_position]
        index_for_node = self.indices['nodes'][node_index]
        new_vector[index_for_node] = player_position + 5
        new_vector[indices_for_hand] -= cost
        return new_vector

    def trade_between_players(self, player_position: int,  card_out_in: tuple, player_accepting_trade: int = None) -> np.array:
        '''
        Trade resources between players. The resources are given as a tuple of indices of the resources to be traded.
        If player_from is None, the resources are only changed for player_position.

        Function returns a new vector as np.array with the updated resources.

        Returns:
            np.array: A new vector with the updated resources after the trade.
        '''
        resources = np.zeros(len(self.players[0].hand), np.float64)
        resources[card_out_in[0]] = -1
        resources[card_out_in[1]] = 1
        new_vector = self.vector.copy()
        indices_for_hand_to = self.indices['hand_for_player'][player_position]
        new_vector[indices_for_hand_to] += resources
        if player_accepting_trade is not None:
            indices_for_hand_from = self.indices['hand_for_player'][player_accepting_trade]
            new_vector[indices_for_hand_from] -= resources
        
        return new_vector
    
    def trade_with_bank(self, player_position: int, card_out_in: tuple) -> np.array:
        '''
        Trade resources with the bank. The resources are given as a tuple of indices of the resources to be traded.
        The player gives 4 of one resource and receives 1 of another resource.

        Function returns a new vector as np.array with the updated resources.

        Returns:
            np.array: A new vector with the updated resources after the trade with the bank.
        '''
        resources = np.zeros(len(self.players[0].hand))
        resources[card_out_in[0]] = -4
        resources[card_out_in[1]] = 1
        new_vector = self.vector.copy()
        indices_for_hand = self.indices['hand_for_player'][player_position]
        new_vector[indices_for_hand] += resources
        return new_vector
    
    def split_vector(self, vector: np.array = None) -> list[np.array]:
        '''
        Split the vector into a list of vectors.
        The elements are 'turns', 'ranks', 'values', 'nodes', 'edges', and 'hands'.
        If vector is None, the current vector of the BoardVector instance is used.
        If a vector is provided, it is used instead of the instance's vector.

        Returns:
            list[np.array]: A list of numpy arrays representing the different parts of the vector.
        '''
        if vector is None:
            vector = self.vector
        vectors = []
        vectors += [vector[self.indices['turns']]]
        vectors += [vector[self.indices['ranks']]]
        vectors += [vector[self.indices['values']]]
        vectors += [vector[self.indices['nodes']]]
        vectors += [vector[self.indices['edges']]]
        vectors += [vector[self.indices['hands']]]

        return vectors

    
    
    