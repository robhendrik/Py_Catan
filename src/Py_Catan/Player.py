from Py_Catan.Preferences import PlayerPreferences
from Py_Catan.BoardStructure import BoardStructure
import numpy as np
import random
from functools import cache

class Player:
    def __init__(self,
                 name: str = 'A', 
                 structure: BoardStructure = BoardStructure(),
                 preference: any = PlayerPreferences(), 
                 status: dict = dict([]) ):
        if not status:
            # attributes at instance creation
            self.name = name
            self.preference = preference
            self.structure = structure

            # board properties, changing during game
            self.free_nodes_on_board = np.ones(self.structure.no_of_nodes,np.int16)
            self.free_edges_on_board = np.ones(self.structure.no_of_edges,np.int16)
            self.earnings_per_node = None
            self.longest_street_on_board = structure.longest_street_minimum
    
            # player properties depending on position in game
            self.hand = np.zeros(self.structure.no_of_resource_types,np.int16)
            self.streets = np.zeros(self.structure.no_of_edges,np.int16)
            self.villages = np.zeros(self.structure.no_of_nodes,np.int16)
            self.towns = np.zeros(self.structure.no_of_nodes,np.int16)
            self.earning_power = np.zeros(self.structure.no_of_resource_types,np.int16)
            self.owns_longest_street = False
            self.longest_street_for_this_player = 0
            self.build_options = {  'street_options':self.streets,
                                    'village_options':self.villages,
                                    'secondary_village_options': self.villages}
            
            #rounding and technical parameters`
            self.atol = 0.00
            self._start_stop_nodes = set([])
            self._already_build_streets = []
            self._actions_in_round = 0
            # used for situations where player needs know the other players, or the order in the game
            self._players_in_this_game = []
        else:
            self.from_player_status(status)

    def copy(self) -> 'Player':
        '''
        Create a copy of the player with the same attributes.
        The name is set to 'New' to avoid conflicts with the original player.
        '''
        new_player = Player('New')
        np = vars(new_player)
        for k,v in vars(self).items():
            if k != 'name':
                try:
                    np[k] = v.copy()
                except:
                    np[k] = v
        return new_player
    
    def calculate_score(self) -> int:
        ''' 
        Calculate the score of the player based on their current state.
        The score is calculated as follows:   
        - Each town is worth 2 points
        - Each village is worth 1 point 
        - If the player has the longest street, they get an additional 2 points
        '''              
        score = 0
        score += sum(self.towns) * 2
        score += sum(self.villages) * 1
        score += 2 if  self.owns_longest_street else 0
        return score
    
    def respond_positive_to_other_players_trading_request(self,card_out_in) -> bool:
        '''        
        Return 'True' if this player responds positively to a trading request from another player.
        The player will accept the trade if they have at least one card of the type they are receiving
        and at least one card of the type they are giving away. 

        In the base class the response is random, with a 50% chance of accepting the trade.
        This can be overridden in subclasses to implement more complex trading strategies.
        '''
        if self.hand[card_out_in[0]] <= 0:
            return False
        seed = random.randint(0,100)
        return seed < 50

    def can_build_street(self) -> bool:
        ''' 
        Check if the player can build a street.
        The player can build a street if they have enough resources to pay for it.
        The cost of a street is defined in the real_estate_cost attribute. 
        '''
        enough_cards = all((self.hand - np.array(self.structure.real_estate_cost[0]) >= 0))
        streets_left = np.sum(self.streets) < self.structure.max_available_real_estate_per_type[0]
        return enough_cards and streets_left
    
    def can_build_village(self) -> bool:
        '''
        Check if the player can build a village.
        The player can build a village if they have enough resources to pay for it.     
        The cost of a village is defined in the real_estate_cost attribute.
        '''
        enough_cards = all((self.hand - np.array(self.structure.real_estate_cost[1]) >= 0))
        villages_left = np.sum(self.villages) < self.structure.max_available_real_estate_per_type[1]
        return enough_cards and villages_left
    
    def can_build_town(self)    -> bool:
        '''
        Check if the player can build a town.   
        The player can build a town if they have enough resources to pay for it.
        The cost of a town is defined in the real_estate_cost attribute.
        '''
        enough_cards = all((self.hand - np.array(self.structure.real_estate_cost[2]) >= 0)) 
        empty_village = np.sum(self.villages) > 0
        towns_left  = np.sum(self.towns) < self.structure.max_available_real_estate_per_type[2]
        return enough_cards and empty_village and towns_left

    def can_trade_with_player(self) -> bool:
        '''
        Check if the player can trade with another player.
        The player can trade with another player if they have at least one card of any type.    
        '''
        return max(self.hand) > 0

    def can_trade_with_specific_player(self) -> bool:
        '''
        Check if the player can trade with a specific player.
        The player can trade with a specific player if they have at least one card of the type they are giving away.
        '''
        return False

    def can_trade_with_bank(self):
        '''
        Check if the player can trade with the bank.
        The player can trade with the bank if they have at least 4 cards of any type.
        '''
        return max(self.hand) >= 4

    def build_street(self, edge) -> None:
        '''
        Build a street on the board.
        The edge must be free and the player must have a village or town on one of the  
        nodes connected to the edge.
        The edge is marked as occupied and the player's street count is updated.
        The longest street on the board is updated for this player.
        '''
        self.hand -= np.array(self.structure.real_estate_cost[0])
        self.streets[edge] = 1
        self.calculate_longest_street()
        return
    
    def build_village(self, node)   -> None:   
        '''
        Build a village on the board.
        The node must be free and the player must have a street on one of the edges connected   
        to the node.
        The node is marked as occupied and the player's village count is updated.
        The player's hand is updated with the resources used to build the village.
        The free nodes on the board are updated.    
        '''
        self.hand -= np.array(self.structure.real_estate_cost[1])
        self.villages[node] = 1
        return 
    
    def build_town(self,node) -> None:
        ''' 
        Build a town on the board.
        The node must be occupied by a village of the player.   
        The player's hand is updated with the resources used to build the town.
        The village is removed from the player's villages and the town is added.
        '''
        self.hand -= np.array(self.structure.real_estate_cost[2])
        self.towns[node] = 1
        self.villages[node] = 0
        return 
    
    def trade_with_bank(self,card_out_in) -> None:
        '''
        Trade with the bank.
        The player gives 4 cards of one type and receives 1 card of another type.   
        '''
        self.hand[card_out_in[0]] -= 4
        self.hand[card_out_in[1]] += 1
        return 
    
    def trade_with_player(self,card_out_in) -> None:
        '''
        Trade with another player.
        The player gives 1 card of one type and receives 1 card of another type.
        '''
        self.hand[card_out_in[0]] -= 1
        self.hand[card_out_in[1]] += 1
        return 
    
    def update_build_options(self) -> None:
        ''' 
        Update the build options for this player.
        The build options are updated based on the current state of
        the player's hand, streets, villages and towns.
        The build options are stored in the build_options attribute as a dictionary.
        '''
        build_options_for_villages = np.logical_and(self.free_nodes_on_board,self.streets @ self.structure.edge_node_matrix )
        build_options_for_streets = np.logical_and(self.free_edges_on_board,self.streets @ self.structure.edge_edge_matrix )

        helper = np.logical_and(np.logical_not(build_options_for_villages),build_options_for_streets @ self.structure.edge_node_matrix)
        build_options_for_streets_to_free_nodes = np.logical_and(self.free_nodes_on_board,helper)

        self.build_options = {'street_options':build_options_for_streets,
                            'village_options':build_options_for_villages,
                            'secondary_village_options': build_options_for_streets_to_free_nodes}
        return 
    
    @cache
    def _algo_longest_street(self, streets_for_calculation):
        already_build_streets = [list(edge) for edge,street in zip(self.structure.nodes_connected_by_edge,streets_for_calculation) if street != 0]
        nodes = []
        for edge in already_build_streets:
            nodes += edge
        start_or_stop_nodes = set([])
        for node in nodes:
            if node not in start_or_stop_nodes:
                start_or_stop_nodes.add(node)
        routes = [{'end':start,'used':[]} for start in start_or_stop_nodes]
        longest = 0
        while routes:
            route = routes.pop(0)
            longest = max(len(route['used']),longest)
            for edge in already_build_streets:
                if edge in route['used']  or route['end'] not in edge:
                    continue
                if edge[0] == route['end']:
                    routes.append({'end':edge[1],'used':route['used'] + [edge]})
                else:
                    routes.append({'end':edge[0],'used':route['used'] + [edge]})
        return longest

    def calculate_longest_street(self, added_edge=None) -> int:
        '''
        Calculate the longest street for this player.   
        The longest street is calculated by finding the longest path in the graph
        defined by the nodes connected by the edges occupied by this player.    
        The longest street is the maximum length of the paths found.
        The paths are found by starting from each node connected by an edge occupied by this player
        and exploring all possible paths until no more edges can be traversed.
        The length of the path is the number of edges traversed.

        self.longest_street_for_this_player is updated with the length of the longest street found.
        '''
        if added_edge is None:
            self.longest_street_for_this_player = self._algo_longest_street(streets_for_calculation=tuple(self.streets))
            return self.longest_street_for_this_player
        else:
            self.longest_street_for_this_player = self._algo_longest_street(added_edge=added_edge) 
            return self.longest_street_for_this_player
    
    def create_player_status(self):
        status = {
            'name': self.name,
            'hand': self.hand.copy(),
            'preference': PlayerPreferences(**self.preference.asdict()) ,
            'streets': self.streets.copy(),
            'villages': self.villages.copy(),
            'towns': self.towns.copy(),
            'earning_power': self.earning_power.copy(),
            'longest_street_for_this_player': self.longest_street_for_this_player,
            'free_nodes_on_board': self.free_nodes_on_board.copy() if self.free_nodes_on_board is not None else None,
            'free_edges_on_board': self.free_edges_on_board.copy() if self.free_edges_on_board is not None else None,
        }
        return status
    
    def from_player_status(self, status):
        self.name = status['name']
        self.preference = status['preference']
        self.hand = status['hand'].copy()
        self.streets = status['streets'].copy()
        self.villages = status['villages'].copy()
        self.towns = status['towns'].copy()
        self.earning_power = status['earning_power'].copy()
        self.longest_street_for_this_player = status['longest_street_for_this_player']
        self.free_nodes_on_board = status['free_nodes_on_board'].copy() if status['free_nodes_on_board'] is not None else None
        self.free_edges_on_board = status['free_edges_on_board'].copy() if status['free_edges_on_board'] is not None else None
        return 
    
    def calculate_value(self):
        return 0

    def create_vector_for_player(self, position: int, include_values: bool = True) -> np.ndarray[np.float32]:
        '''
        Create a vector representation of the player for use in machine learning models.
        The vector is created based on the player's hand, streets, villages, towns, and other attributes.
        
        Default the vector for nodes and edges is created for a player at 'position 0' on the board,
        so edges with a street are set to 1, and nodes with a village are set to 1 and nodes with a town are set to 5.

        If position is not 0, the vector is created for the player at that position.

        returns:
            np.ndarray: A numpy array representing the player.
        '''
        vector = np.zeros(len(self.structure.header), np.float32)
        # === value ===
        if include_values:
            values_indices = self.structure.vector_indices['values']
            vector[values_indices[position]] = self.calculate_value()
        # === hand ===
        hand = self.structure.vector_indices['hand_for_player'][position]
        vector[hand] = self.hand
        # === edges ===
        edges = self.structure.vector_indices['edges']
        vector[edges] = (1+position)*self.streets
        # === nodes ===
        nodes = self.structure.vector_indices['nodes']
        vector[nodes] = (1+position)*self.villages + (5+position)*self.towns
        return vector.astype(np.float32)

 
    
    
    

    
 