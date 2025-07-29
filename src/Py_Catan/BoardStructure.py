import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass, field, asdict
from Py_Catan.BoardLayout import BoardLayout
    
class BoardStructure:
    """
    BoardStructure
    Class to define the structure and logic of a Catan-like board game.
    It initializes the board based on the provided BoardLayout.                 
    It calculates the coordinates for nodes, edges, and tiles,
    as well as the relationships between them.
                        
    It also calculates the dice impact per node and the earning power of each node.
    It provides methods to convert polar coordinates to node, tile, and edge indices.
    It generates a list of all possible board configurations based on the tile layout.

    Attributes:
        values (list): Copy of dice values from the board layout.
        tile_layout (str): String representing the resource type of each tile.
        _scale (float): Scale factor for the board.
        _rings (int): Number of rings (layers) in the board.
        street_cost (str): Cost of building a street.
        village_cost (str): Cost of building a village.
        town_cost (str): Cost of building a town.
        development_card_cost (str): Cost of a development card.
        winning_score (int): Score required to win the game.
        dice_value_to_hand_in_cards (dict): Mapping of dice values to card hand-ins.
        max_available_real_estate_per_type (list): Maximum available real estate per type.
        longest_street_minimum (int): Minimum length for the longest street.
        vectors (list): List of numpy arrays representing hex grid directions.
        _node_coordinates (list): List of node coordinates.
        _edge_coordinates (list): List of edge coordinates.
        _tile_coordinates (list): List of tile coordinates.
        neighbour_nodes_for_tiles (list): List of sets of neighbouring nodes for each tile.
        nodes_connected_by_edge (list): List of sets of nodes connected by each edge.
        neighbour_nodes_for_nodes (list): List of sets of neighbouring nodes for each node.
        secondary_neighbour_nodes_for_nodes (list): List of sets of secondary neighbours for each node.
        resource_types (str): String of unique resource types.
        dice_results (list): List of unique dice results.
        dice_impact_per_node_dnt (np.ndarray): Dice impact per node for each resource type.
        node_earning_power (np.ndarray): Earning power of each node.
        no_of_nodes (int): Number of nodes.
        no_of_edges (int): Number of edges.
        no_of_resource_types (int): Number of resource types.
        real_estate_cost (list): Resource cost for each real estate type.
        node_neighbour_matrix (np.ndarray): Matrix indicating node adjacency.
        edge_edge_matrix (np.ndarray): Matrix indicating edge adjacency.
        edge_node_matrix (np.ndarray): Matrix indicating which nodes are connected by each edge.
        board_settings_for_players (dict): Board settings for use by players.
    Methods:
        __init__(self, board_layout: BoardLayout = BoardLayout()):
            Initializes the board structure and computes all relationships and matrices.
        _calc_coordinates(self):
            # Calculates the coordinates for nodes, edges, and tiles on the board.
        _calc_nodes_connected_by_edge(self):
            # Determines which nodes are connected by each edge.
        _calc_neighbour_nodes_for_nodes(self):
            # Determines direct neighbouring nodes for each node.
        _calc_secondary_neighbour_nodes_for_nodes(self):
            # Determines secondary (two-step) neighbours for each node.
        _calc_neighbour_nodes_for_tiles(self):
            # Determines which nodes are adjacent to each tile.
        _calc_dice_impact_per_node_dnt(self) -> np.ndarray:
            # Calculates the dice impact per node for each resource type.
        _calc_node_earning_power(self) -> np.ndarray:
            # Calculates the earning power of each node based on dice probabilities.
        polar_to_node(self, polar) -> int:
            # Converts polar coordinates to a node index.
        polar_to_tile(self, polar) -> int:
            # Converts polar coordinates to a tile index.
        polar_to_edge(self, polar) -> int:
            # Converts polar coordinates to an edge index.
        generate_list_of_all_possible_boards(self) -> list:
            # Generates all unique board configurations based on tile layout constraints.
        logging_header(self) -> list[str]:
            Create a header for logging the board state.
        """


    def __init__(self, board_layout: BoardLayout = BoardLayout()) -> None:
        ''' 
        Initializes the board structure and computes all relationships and matrices.
        
        Args:   
                board_layout (BoardLayout): An instance of BoardLayout containing the board configuration.
        Returns:        
                None    
        '''
        # setting up the board structure based on the layout
        self.values = board_layout.values.copy()
        self.tile_layout =  board_layout.tile_layout
        self._scale = board_layout.scale
        self._rings = board_layout.rings
        self.resource_types = board_layout.resource_types
        self.street_cost = board_layout.street_cost
        self.village_cost = board_layout.village_cost
        self.town_cost = board_layout.town_cost
        self.development_card_cost = board_layout.development_card_cost
        self.winning_score = board_layout.winning_score
        self.dice_value_to_hand_in_cards = board_layout.dice_value_to_hand_in_cards
        self.max_available_real_estate_per_type = [
            board_layout.max_available_streets,
            board_layout.max_available_villages,
            board_layout.max_available_towns
        ]
        self.longest_street_minimum = board_layout.longest_street_minimum
        self.plot_colors_players = board_layout.plot_colors_players
        self.plot_labels_for_resources = board_layout.plot_labels_for_resources
        self.plot_max_card_in_hand_per_type = board_layout.plot_max_card_in_hand_per_type

        # setting up the vectors for the hexagonal grid
        self.vectors = [
                np.array([1.0*self._scale,0],np.float16),
                np.array([0.5*self._scale,-0.5*self._scale*np.sqrt(3)],np.float16),
                np.array([-0.5*self._scale,-0.5*self._scale*np.sqrt(3)],np.float16),
                np.array([-1.0*self._scale,0],np.float16),
                np.array([-0.5*self._scale,0.5*self._scale*np.sqrt(3)],np.float16),
                np.array([0.5*self._scale,0.5*self._scale*np.sqrt(3)],np.float16),
            ]
        
        # calculating the coordinates for nodes, edges and tiles
        # as well as the relationships between them
        self._node_coordinates,self._edge_coordinates,self._tile_coordinates = self._calc_coordinates()
        self.neighbour_nodes_for_tiles = self._calc_neighbour_nodes_for_tiles()
        self.nodes_connected_by_edge = self._calc_nodes_connected_by_edge()
        self.neighbour_nodes_for_nodes = self._calc_neighbour_nodes_for_nodes()
        self.secondary_neighbour_nodes_for_nodes = self._calc_secondary_neighbour_nodes_for_nodes()
        self.dice_results = list(sorted(set(self.values)))
        self.dice_impact_per_node_dnt = self._calc_dice_impact_per_node_dnt()
        self.node_earning_power = self._calc_node_earning_power()
        
        self.no_of_nodes = len(self._node_coordinates)
        self.no_of_edges = len(self._edge_coordinates)
        self.no_of_resource_types = len(self.resource_types)
        self.real_estate_cost = [
            tuple(self.street_cost.count(c) for c in board_layout.resource_types),
            tuple(self.village_cost.count(c) for c in board_layout.resource_types),
            tuple(self.town_cost.count(c) for c in board_layout.resource_types),
            tuple(self.development_card_cost.count(c) for c in board_layout.resource_types)
        ]
        
        self.node_neighbour_matrix = np.array([
            [1 if (c==r or c in self.neighbour_nodes_for_nodes[r]) else 0 for c in range(self.no_of_nodes)]
            for r in range(self.no_of_nodes)],dtype=np.int8)
        
        def helper(r,c):
            a = self.nodes_connected_by_edge[r]
            b = self.nodes_connected_by_edge[c]
            return len(set(a).intersection(b)) > 0 and set(a) != set(b)
        
        self.edge_edge_matrix = np.array([
            [1 if helper(r,c) else 0 for c in range(self.no_of_edges)]
            for r in range(self.no_of_edges)],dtype=np.int8)
        
        self.edge_node_matrix = np.array([
            [1 if n in self.nodes_connected_by_edge[e] else 0 for n in range(self.no_of_nodes)]
            for e in range(self.no_of_edges)],dtype=np.int8)
        
        # ===== Set paramaters for creating board vector, used for 
        # training Keras model and for logging board status =====
        self.number_of_players_for_logging = 4 # default for creating log to train AI
        self.header = self.logging_header()
        self.vector_indices = self.get_indices()
        
        
    def _calc_coordinates(self):
        node_coordinates = []
        tile_coordinates = []
        edge_coordinates = []
        for r in range(self._rings):
            nodes_in_ring = []
            i = 0
            nodes_in_ring.append((1+r)*self.vectors[(4+i)%6] + r*self.vectors[(5+i)%6])
            for _ in range(6):
                nodes_in_ring.append(nodes_in_ring[-1] + self.vectors[i%6])
                for _ in range(r):
                    nodes_in_ring.append(nodes_in_ring[-1] + self.vectors[(i+1)%6])
                    nodes_in_ring.append(nodes_in_ring[-1] + self.vectors[i%6])
                i+=1
            nodes_in_ring.pop(-1)
      
            i = 0
            tiles_in_ring = []
            if r==0:
                tiles_in_ring.append(nodes_in_ring[0]+self.vectors[1] )
            else:
                module = 3+2*(r-1) # number of nodes in ring divided by 6
                for index in range(len(nodes_in_ring)):
                    if index % module == 0:
                        i+=1
                        tiles_in_ring.append(nodes_in_ring[index]+self.vectors[(i)%6] )
                        for step in range(1,r):  
                            tiles_in_ring.append(nodes_in_ring[index+step*2]+self.vectors[(i)%6] ) 
  
            edges_in_ring = []
            for node_number,node in enumerate(nodes_in_ring):
                edges_in_ring.append([node, nodes_in_ring[(node_number+1)%len(nodes_in_ring)]])
        
            if r < self._rings-1:
                module = 3+2*(r-1) # number of nodes in ring divided by 6
                i = 0
                for index in range(len(nodes_in_ring)):
                    if index % module == 0:
                        edges_in_ring.append([nodes_in_ring[index],nodes_in_ring[index]+self.vectors[(i+4)%6]] )
                        i+=1
                        for step in range(r):
                            edges_in_ring.append([nodes_in_ring[index+1+step*2],nodes_in_ring[index+1+step*2]+self.vectors[(i+4)%6]] )

            node_coordinates = node_coordinates + nodes_in_ring
            edge_coordinates = edge_coordinates + edges_in_ring
            tile_coordinates = tile_coordinates + tiles_in_ring
        return node_coordinates,edge_coordinates,tile_coordinates
  
    def _calc_nodes_connected_by_edge(self) -> list:
        '''
        Calculates the nodes connected by each edge in the hexagonal grid.
        Each edge connects two nodes, and this method returns a list of sets,       
        where each set contains the indices of the two nodes connected by that edge.
        The edges are calculated based on the number of rings in the hexagonal grid.
        The first ring has 6 edges, the second ring has 12 edges, and so on.
        The number of edges in each ring is given by the formula: 6 * (r + 1) for r >= 0.
        Returns:
            list: A list of sets, where each set contains two node indices connected by an edge.'''
        nodes_connected_by_edge = []
        for r in range(self._rings):
            node_this_ring = 6*r*r # 0,6,24,54 ... #18 +12(r-1) = 12r + 6, som = 6R + 6R(R-1) =6R^2 
            node_next_ring = 6*(r+1)*(r+1)
            for node_number in range(node_this_ring,node_next_ring-1):
                nodes_connected_by_edge.append(set([node_number, node_number+1]))
            nodes_connected_by_edge.append(set([6*r*r,6*(r+1)*(r+1)-1]))
            if r < self._rings-1:
                module = 3+2*(r-1) # number of nodes in ring divided by 6
                for index,node_number in enumerate(range(node_this_ring,node_next_ring)):
                    if index % module == 0:
                        if node_next_ring == 6*(r+1)*(r+1):
                            nodes_connected_by_edge.append(set([node_this_ring,6*(r+2)*(r+2)-1]) )
                            node_this_ring += 1
                            node_next_ring += 2
                        else:
                            nodes_connected_by_edge.append(set([node_this_ring,node_next_ring]) )
                            node_this_ring += 1
                            node_next_ring += 3
                        for step in range(r):
                            nodes_connected_by_edge.append(set([node_this_ring,node_next_ring]) )
                            node_this_ring += 2
                            node_next_ring += 2
        return nodes_connected_by_edge
  
    def _calc_neighbour_nodes_for_nodes(self):
        neighbour_nodes_for_nodes = [set([]) for _ in range(len(self._node_coordinates))]
        for edge in self.nodes_connected_by_edge:
            edge = list(edge)
            neighbour_nodes_for_nodes[edge[0]].add(edge[1])
            neighbour_nodes_for_nodes[edge[1]].add(edge[0])
        return neighbour_nodes_for_nodes

    def _calc_secondary_neighbour_nodes_for_nodes(self):
        secondary_neighbour_nodes_for_nodes = [set([]) for _ in range(len(self._node_coordinates))]
        for node,direct_neighbours in enumerate(self.neighbour_nodes_for_nodes):
            for nb in direct_neighbours:
                for secondary_connection in self.neighbour_nodes_for_nodes[nb]:
                    if secondary_connection not in direct_neighbours and secondary_connection is not node:
                        secondary_neighbour_nodes_for_nodes[node].add(secondary_connection)
        return secondary_neighbour_nodes_for_nodes
    
    def _calc_neighbour_nodes_for_tiles(self):
        neighbour_nodes_for_tiles = []
        for r in range(self._rings):
            neighbour_nodes_per_tile = []
            if r==0:
                neighbour_nodes_per_tile.append( set([n for n in range(6)]) )
            else:
                node_ring_above = 6*r*r # 0,6,24,54 ... #18 +12(r-1) = 12r + 6, som = 6R + 6R(R-1) =6R^2 
                node_ring_below = 6*(r-1)*(r-1)
                for _ in range(6):
                    neighbour_nodes_below = [ node_ring_below + n for n in range(2)]
                    if neighbour_nodes_below[-1] == 6*r*r:
                        neighbour_nodes_below[-1] = 6*(r-1)*(r-1)
                    if node_ring_above == 6*r*r:
                        neighbour_nodes_above =  [node_ring_above + (12*r+6) -1] +  [(node_ring_above + n)  for n in range(0,3)] 
                    else:
                        neighbour_nodes_above =  [(node_ring_above + n)  for n in range(0,4)] 
                    neighbour_nodes_per_tile.append(set(neighbour_nodes_below + neighbour_nodes_above))
                    node_ring_below, node_ring_above =  neighbour_nodes_below[-1],neighbour_nodes_above[-1]
                    for _ in range(1,r):
                        neighbour_nodes_below = [ node_ring_below + n for n in range(3)]
                        neighbour_nodes_above =  [node_ring_above + n for n in range(3)] 
                        node_ring_below, node_ring_above =  neighbour_nodes_below[-1],neighbour_nodes_above[-1]
                        if neighbour_nodes_below[-1] == 6*r*r:
                            neighbour_nodes_below[-1] = 6*(r-1)*(r-1)
                        neighbour_nodes_per_tile.append(set(neighbour_nodes_below + neighbour_nodes_above))
            neighbour_nodes_for_tiles = neighbour_nodes_for_tiles + neighbour_nodes_per_tile
        return neighbour_nodes_for_tiles

    def _calc_dice_impact_per_node_dnt(self) -> np.ndarray  :
        '''
        Calculate the dice impact per node for each resource type.
        The dice impact is calculated by iterating over all tiles and for each tile,
        iterating over all its neighbour nodes.
        The dice impact is stored in a numpy array with shape (no_of_dice_results, no_of_nodes, no_of_resource_types).
        The first dimension corresponds to the dice results (1-6),
        the second dimension corresponds to the nodes, and the third dimension corresponds to the resource types.
        The dice impact is calculated by counting how many times each resource type is present in the neighbour nodes
        for each dice result.
        The method returns the dice impact per node as a numpy array.
        '''
        dice_impact_per_node_dnt = np.zeros((len(self.dice_results),len(self._node_coordinates),len(self.resource_types)),dtype=np.int16)
        for tile in range(len(self._tile_coordinates)):
            resource = self.resource_types.index(self.tile_layout[tile])
            dice_result = self.dice_results.index(self.values[tile])
            for nb in self.neighbour_nodes_for_tiles[tile]:
                dice_impact_per_node_dnt[dice_result,nb,resource] += 1
        return dice_impact_per_node_dnt
    
    def _calc_node_earning_power(self) -> np.ndarray:
        '''
        Calculate the earning power of each node based on the dice results.
        The earning power is calculated by summing the dice impact for each node
        for all possible dice results (1-6) and excluding the result of 7.
        The earning power is stored in a numpy array with shape (no_of_nodes, no_of_resource_types).
        The earning power is calculated by iterating over all nodes and for each node,
        iterating over all possible dice results (1-6) and summing the dice impact  
        for each node and each resource type.
        '''
        node_earning_power = np.zeros((len(self._node_coordinates),len(self.resource_types)),dtype=np.int16)
        for node in range(len(self._node_coordinates)):
            for dice_1 in range(1,7):
                for dice_2 in range(1,7):
                    if dice_1 + dice_2 == 7:
                        continue
                    dice_result = self.dice_results.index(dice_1 + dice_2)
                    node_earning_power[node] += self.dice_impact_per_node_dnt[dice_result,node]
        return node_earning_power
    
    def polar_to_node(self, polar) -> int:
        '''
        Convert polar coordinates to node index.
        The node index is calculated based on the polar coordinates.    
        The formula used is:
        node_index = 3*polar[0]*(polar[0]-1) + polar[1]
        where polar[0] is the ring number and polar[1] is the position in the ring.
        This formula is derived from the fact that each ring has 6 nodes and the nodes are  
        numbered in a clockwise direction starting from the top node of the first ring.
        The first ring has 6 nodes, the second ring has 12 nodes, the third
        ring has 18 nodes, and so on. The node index is calculated by multiplying the ring number
        by 3 and adding the position in the ring.'''
        return 6*polar[0]*polar[0] + polar[1]

    def polar_to_tile(self, polar) -> int:
        '''
        Convert polar coordinates to tile index.
        The tile index is calculated based on the polar coordinates.
        The formula used is:
        tile_index = 3*polar[0]*(polar[0]-1) + polar[1] + 1
        where polar[0] is the ring number and polar[1] is the position in the ring.
        This formula is derived from the fact that each ring has 6 tiles and the tiles are      
        numbered in a clockwise direction starting from the top tile of the first ring.
        The first ring has 6 tiles, the second ring has 12 tiles, the third
        ring has 18 tiles, and so on. The tile index is calculated by multiplying the ring number
        by 3 and adding the position in the ring, and then adding 1 to account for the first tile.'''
        if polar == (0,0):
            return 0
        else:
            return 3*polar[0]*(polar[0]-1) + polar[1] + 1
        
    def polar_to_edge(self, polar) -> int:
        ''' 
        Convert polar coordinates to edge index.
        The edge index is calculated based on the polar coordinates.

        The formula used is:
        edge_index = 6*polar[0]*polar[0] + polar[1] + 6*polar[0] - 1
        where polar[0] is the ring number and polar[1] is the position in the ring.
        This formula is derived from the fact that each ring has 6 edges and the edges are      
        numbered in a clockwise direction starting from the top edge of the first ring.
        The first ring has 6 edges, the second ring has 12 edges, the third
        ring has 18 edges, and so on. The edge index is calculated by multiplying the ring number
        by 6 and adding the position in the ring, and then adding 6 times the 
        ring number minus 1 to account for the edges in the previous rings.

        '''
        #24 + 18(r-1) = 6+18r
        #[6,24,42] ->[6,30,72]
        #6r + 9r(r-1) = 0,6,30,72 = 9*r*r - 3r = 0,6,30
        return 9*polar[0]*polar[0] - 3*polar[0] + polar[1]
 
    def generate_list_of_all_possible_boards(self) -> list:
        ''' 
        Generate a list of all possible board configurations based on the tile layout.
        This method generates all unique combinations of tiles for the first ring,
        ensuring that no tile is repeated in the same position and that the first and last tiles are different.
        It then adds a second row of tiles, ensuring that the same rules apply for the second row as well.
        The method returns a list of all unique board configurations as strings, where each string represents a board configuration.
        The first character of the string is 'D' for desert, followed by the tiles in the first row,
        and then the tiles in the second row.   
        This method is useful for generating all possible board configurations for the game of Catan.

        NOTE: No yet fully using the settings in layout, this is on TODO list. It will only generate boards
        with 2 rings (standard Catan board) and tiles from 'SWGOB' plus desert in the centre.
        '''
        #For first ring number of combinations:
        # sequence of 6 without same twice, including closing circle
        tiles = {'S':4,'W':4,'G':4, 'O': 3, 'B':3}
        boards = [tile for tile in tiles.keys()]
        for _ in range(5):
            new_boards = []
            for board in boards:
                for tile in tiles.keys():
                    if len(board) < 5:
                        if tile != board[-1] and board.count(tile) < tiles[tile]:
                            new_boards.append(board + tile)
                    else:
                        if tile != board[-1] and tile != board[0] and board.count(tile) < tiles[tile]:
                            new_boards.append(board + tile)
            boards = new_boards

        # only unique permutations
        def permutations(s):
            return [s[n:] + s[:n] for n in range(len(s))]

        uniques = []
        for board in boards:
            for p in permutations(board):
                if p in uniques:
                    break
            else:
                uniques.append(board)

        # add second row
        full_boards = []
        for p in uniques:
            for index in range(12):
                if index == 0 and p.count(tile) < tiles[tile]:
                    rings = [tile for tile in tiles if tile != p[0]]
                elif index in [1,3,7,9]:
                    new_rings = []
                    for ring in rings:
                        for tile in tiles:
                            if tile == ring[-1]:
                                continue
                            if tile == p[index//2] or tile == p[(index//2) + 1]:
                                continue
                            if ring.count(tile) + p.count(tile) >= tiles[tile]:
                                continue
                            new_rings.append(ring + tile)
                    rings = new_rings
                elif index in [2,4,6,8]:
                    new_rings = []
                    for ring in rings:
                        for tile in tiles:
                            if tile == ring[-1]:
                                continue
                            if tile == p[index//2]:
                                continue
                            if ring.count(tile) + p.count(tile) >= tiles[tile]:
                                continue
                            new_rings.append(ring + tile)
                    rings = new_rings
                else:
                    new_rings = []
                    for ring in rings:
                        for tile in tiles:
                            if tile == ring[0] or tile == ring[-1]:
                                continue
                            if tile == p[0] or tile == p[-1]:
                                continue
                            if ring.count(tile) + p.count(tile) >= tiles[tile]:
                                continue
                            new_rings.append(ring + tile)
                    rings = new_rings
            for ring in rings:
                full_boards.append( ("D",p,ring))

        return [b[0] + b[1] + b[2] for b in full_boards]
    
    def logging_header(self) -> list[str]:
        '''
        Create a header for logging the board state.
        '''
        headers = [
            'turns_before_end',
            'rank_A',
            'rank_B',
            'rank_C',
            'rank_D',
            'value_A',
            'value_B',
            'value_C',
            'value_D'
        ]
        headers += ['node_'+str(n) for n in range(self.no_of_nodes)]
        headers += ['egde_'+str(n) for n in range(self.no_of_edges)]
        headers += ['hand_A_'+str(n) for n in range(self.no_of_resource_types)]
        headers += ['hand_B_'+str(n) for n in range(self.no_of_resource_types)]
        headers += ['hand_C_'+str(n) for n in range(self.no_of_resource_types)]
        headers += ['hand_D_'+str(n) for n in range(self.no_of_resource_types)]
        return headers
    
    def get_indices(self) -> dict:
        '''
        Create a dictionary with the indices of the attributes in the vector.
        The indices are used to access the attributes in the vector.

        Returns:
            dict: A dictionary with the indices of the attributes in the vector.
        '''
        indices = {
            'turns_before_end': 0,
            'rank_A': 1,
            'rank_B': 2,
            'rank_C': 3,
            'rank_D': 4,
            'value_A': 5,
            'value_B': 6,
            'value_C': 7,
            'value_D': 8
        }
        for i in range(self.no_of_nodes):
            indices['node_'+str(i)] = i + 9
        for i in range(self.no_of_edges):
            indices['edge_'+str(i)] = i + 9 + self.no_of_nodes
        for i in range(self.no_of_resource_types):
            indices['hand_A_'+str(i)] = i + 9 + self.no_of_nodes + self.no_of_edges
        for i in range(self.no_of_resource_types):
            indices['hand_B_'+str(i)] = i + 9 + self.no_of_nodes + self.no_of_edges + self.no_of_resource_types
        for i in range(self.no_of_resource_types):
            indices['hand_C_'+str(i)] = i + 9 + self.no_of_nodes + self.no_of_edges + self.no_of_resource_types + self.no_of_resource_types
        for i in range(self.no_of_resource_types):
            indices['hand_D_'+str(i)] = i + 9 + self.no_of_nodes + self.no_of_edges + self.no_of_resource_types + self.no_of_resource_types + self.no_of_resource_types
        indices['turns'] = [indices['turns_before_end']]
        indices['ranks'] = [indices['rank_A'], indices['rank_B'], indices['rank_C'], indices['rank_D']]
        indices['values'] = [indices['value_A'], indices['value_B'], indices['value_C'], indices['value_D']]
        indices['nodes'] = [indices['node_'+str(i)] for i in range(self.no_of_nodes)]
        indices['edges'] = [indices['edge_'+str(i)] for i in range(self.no_of_edges)]
        indices['hands'] = [indices[f'hand_{chr(65+i)}_{j}'] for i in range(self.number_of_players_for_logging) for j in range(self.no_of_resource_types)]
        indices['hand_for_player'] = [[indices[f'hand_{chr(65+i)}_{j}'] for j in range(self.no_of_resource_types)] for i in range(self.number_of_players_for_logging) ]
        return indices