from Py_Catan.Preferences import PlayerPreferences
from Py_Catan.BoardStructure import BoardStructure
import numpy as np
import random
from functools import cache
import warnings

class Player:
    def __init__(self,
                 name: str = 'A', 
                 structure: BoardStructure = BoardStructure(),
                 preference: any = None, 
                 status: dict = dict([]) ):
        """
        Create a new instance of a player. This is the parent class, players with specific behavior
        are implemented as children.

        When creating a player only the 'BoardStructure' is known. So we know how many tiles etc the board has,
        and where the different resources are. When a specific board for a game (Board) is created a list of players is added.
        During the game the player needs to know the status of the board and other players, for this reason information like
        player._players_in_this_game or player_build_options is filled after instance creation, when a player is added to a specific
        board for a game.

        Args:
            name (str, optional): Defaults to 'A'.
            structure (BoardStructure, optional): Defaults to BoardStructure().
            preference (any, optional): _description_. Defaults to PlayerPreferences().
            status (dict, optional): Used to retreive a player from a saved status. Defaults to dict([]).
        """
        if not status:
            # attributes at instance creation
            self.name = name
            if preference is not None:
                warnings.warn("Preference is not used in the base Player class, it is used in child classes. This attribute will be removed.", DeprecationWarning)
                self.preference = preference
            self.structure = structure

            # board properties, changing during game
            self.free_nodes_on_board = np.ones(self.structure.no_of_nodes,np.int16)
            self.free_edges_on_board = np.ones(self.structure.no_of_edges,np.int16)
            self.earnings_per_node = None
            self.longest_street_on_board = structure.longest_street_minimum
    
            # player properties depending on status of the game
            self.hand = np.zeros(self.structure.no_of_resource_types,np.int16)
            self.streets = np.zeros(self.structure.no_of_edges,np.int16)
            self.villages = np.zeros(self.structure.no_of_nodes,np.int16)
            self.towns = np.zeros(self.structure.no_of_nodes,np.int16)
            # potentially obsolete parameter!
            self.earning_power = np.zeros(self.structure.no_of_resource_types,np.int16)
            self.owns_longest_street = False
            self.longest_street_for_this_player = 0

            # build options will to indicated where
            # the player can build streets, villages or towns
            # ensure to first update free_nodes/edges_on_board based
            # on overall board status
            self.build_options = {  'street_options':self.streets,
                                    'village_options':self.villages,
                                    'secondary_village_options': self.villages}
            
            #rounding and technical parameters
            self.atol = 0.00
            self._start_stop_nodes = set([])
            self._already_build_streets = []
            self._actions_in_round = 0

            # used for situations where player needs know the other players or the board, or the order in the game
            # typically this is updated at the beginning of the game, after the board has been created
            self._players_in_this_game = []
            self._board = None
            self._player_position = None
            self._longest_street_has_been_updated = False
        else:
            self.from_player_status(status)

    def copy(self) -> 'Player':
        '''
        Create a copy of the player with the same attributes.
        The name is set to 'New' to avoid conflicts with the original player.

        ==== OVERWRITE THIS IN CHILD CLASSES ====
        The copy function should be overwritten in child classes to ensure that all attributes are copied correctly.
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
    
    def copy_position_from_other_player(self,other_player) -> None:
        """
        Copy the position of other player to self. 
        
        This copies the hand, streets, villages, towns, longest street and other attributes from another player to this
        player without creating a new instance. Specifically player preferences and dedicated attributes from child class are 
        not copied.
        """
        self.hand = other_player.hand.copy()
        self.streets = other_player.streets.copy()
        self.towns = other_player.towns.copy()
        self.villages = other_player.villages.copy()
        self.owns_longest_street = other_player.owns_longest_street
        self.longest_street_for_this_player = other_player.longest_street_for_this_player
        return
    
    def find_best_action(self, 
                         rejected_trades_for_this_round: set = set([]),
                         rejected_trades_for_this_round_for_specific_player: set = set([])
                         ) -> tuple:
        """
        Return a single action as tuple e.g. ('street',0) or ('trade_player',(1,0))
        The implementation of a specific player as child of the generic player class should implement this
        function. This is where the 'intelligence' of a player comes in.
        
        The action types are:
        - 'street': build a street on the given edge 
        - 'village': build a village on the given node
        - 'town': build a town on the given node
        - 'trade_player': trade with another player
        - 'trade_specific_player': trade with a specific player
        - 'trade_bank': trade with the bank
        -   None: do not take any action, pass to next player

        Args:
            rejected_trades_for_this_round (set, optional). Defaults to set([]).
            rejected_trades_for_this_round_for_specific_player (set, optional). Defaults to set([]).

        Returns:
            tuple: action. 

        === OVERWRITE THIS IN CHILD CLASSES ===
        The function should be overwritten in child classes to implement specific player behavior.
        """
        # === maximize the number of actions to avoid endless proposals for trading ===
        if self._actions_in_round >= self.max_actions_in_round:
            return (None,None)
        # === generate random actions, overwrite this to generate player dynamics ===
        action = self.random_action(  rejected_trades_for_this_round=rejected_trades_for_this_round,
                                    rejected_trades_for_specific_player=rejected_trades_for_this_round_for_specific_player
                                    )
        # === example implementation based on value ===
        initial_value = self.calculate_value()
        possible_actions = self.generate_list_of_possible_actions(rejected_trades=rejected_trades_for_this_round)
        if possible_actions:
            # player behavior is based on how value is calculated
            values = self.generate_values_for_possible_actions(possible_actions)[..., self._player_position]
            # in this case 'algorithm' is to select the action with the highest value, could be different.
            best_index = np.argmax(values)
        if not possible_actions or values[best_index] <= initial_value:
            best_action = (None, None)  # No action is better than the current state
        else:
            best_action = possible_actions[best_index]

        return action
    
    def player_setup(self,brd) -> None:
        '''
        Setup the player with initial buildings. Returns a list of two actions to perform during setup 
        (one village and one street).

        The specific implementation of the player should override this method to implement.

        === OVERWRITE THIS IN CHILD CLASSES ===
        The function should be overwritten in child classes to implement specific player behavior during setup.
        '''
        self.update_build_options()
        self.hand = np.array(brd.structure.real_estate_cost[1])
        options = self.generate_possible_actions_for_building_village(set_up=True)
        if not options:
            raise Exception('No options for building village during setup')
        random_index = random.randint(0,len(options)-1)
        random_node = options[random_index][1]
        self.update_build_options()
        self.hand += np.array(brd.structure.real_estate_cost[0])
        options = self.generate_possible_actions_for_building_street(set_up=True, set_up_node=random_node)
        if not options:
            raise Exception('No options for building street during setup')
        random_index = random.randint(0,len(options)-1)
        random_edge = options[random_index][1]
        return [('village', random_node),('street',random_edge)]
    
    def respond_positive_to_other_players_trading_request(self,card_out_in) -> bool:
        '''        
        Return 'True' if this player responds positively to a trading request from another player.
        The player will accept the trade if they have at least one card of the type they are receiving
        and at least one card of the type they are giving away. 

        In the base class the response is random, with a 50% chance of accepting the trade.
        This can be overridden in subclasses to implement more complex trading strategies.

        === OVERWRITE THIS IN CHILD CLASSES ===
        The function should be overwritten in child classes to implement specific player behavior for trading.
        '''
        if self.hand[card_out_in[0]] <= 0:
            return False
        seed = random.randint(0,100)
        return seed < 50
    
    def calculate_value(self):
        """
        Calculate the value of the player. This can be used in gameplay. In this parent class the value is always 0.
        We include this in the parent class since the value is logged in the BoardVector. If the function is not used
        in the functions to generate actions in set_up or during gameplay it will not affect the behavior.
        (incorporating this in parent class is inconsistent, to be fixed in future versions maybe.)

        Returns:
            np.float32: value for the player on the board
        """
        return 0.0
    
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
        The edge is marked as occupied and the player's street count is updated.
        The longest street on the board is updated for this player.

        Updating impact on other players is done on board level, so we do not do it here.
        '''
        self.hand -= np.array(self.structure.real_estate_cost[0])
        self.streets[edge] = 1
        self.calculate_longest_street()
        # Impact on other players should be covered on board level, but we do it here to be sure
        return
    
    def build_village(self, node)   -> None:   
        '''
        Build a village on the board.
        The node must be free and the player must have a street on one of the edges connected   
        to the node. This is not checked in this function but has to be checked before.

        The node is marked as occupied and the player's village count is updated.
        The player's hand is updated with the resources used to build the village.
        
        Updating impact on other players is done on board level, so we do not do it here.
        '''
        self.hand -= np.array(self.structure.real_estate_cost[1])
        self.villages[node] = 1
        return

    def build_town(self,node) -> None:
        ''' 
        Build a town on the board.
        The node must be occupied by a village of the player. This is not checked
        in this function but has to be checked before.

        The player's hand is updated with the resources used to build the town.
        The village is removed from the player's villages and the town is added.

        Updating impact on other players is done on board level, so we do not do it here.
        '''
        self.hand -= np.array(self.structure.real_estate_cost[2])
        self.towns[node] = 1
        self.villages[node] = 0
        return

    
    def trade_with_bank(self,card_out_in) -> None:
        '''
        Trade with the bank.

        Function does not check validity of the action.

        The player gives 4 cards of one type and receives 1 card of another type.   
        '''
        self.hand[card_out_in[0]] -= 4
        self.hand[card_out_in[1]] += 1
        return 
    
    def trade_with_player(self,card_out_in) -> None:
        '''
        Trade with another player.

        Function does not check validity of the action.

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

        NOTE: Always call board._update_board_for_players() before updating build options.
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
    def _algo_longest_street(self, streets_for_calculation: tuple) -> int:
        """
        Algorithm to calculate longest street.
        The longest street is calculated by finding the longest path in the graph
        defined by the nodes connected by the edges occupied by this player.    
        The longest street is the maximum length of the paths found.
        The paths are found by starting from each node connected by an edge occupied by this player
        and exploring all possible paths until no more edges can be traversed.
        The length of the path is the number of edges traversed.

        Args:
            streets_for_calculation (tuple): streets in posession of this player, typically tuple(player.streets)

        Returns:
            int: number of streets in longest street
        """
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
        Calculate the longest street for this player. Also player.longest_street_for_this_player
        is updated with the length of the longest street found.

        self._longest_street_has_been_updated will be set to True to indicate that the longest street has been updated.

        Returns:
            int: number of streets in longest street
        '''
        self.longest_street_for_this_player = self._algo_longest_street(streets_for_calculation=tuple(self.streets))
        self._longest_street_has_been_updated = True
        return self.longest_street_for_this_player
 
    
    def create_player_status(self) -> dict:
        """
        Creates dict with status of the player

        Returns:
            dict: status of the player
        """
        status = {
            'name': self.name,
            'hand': self.hand.copy(),
            'preference': PlayerPreferences(**self.preference.asdict()) ,
            'streets': self.streets.copy(),
            'villages': self.villages.copy(),
            'towns': self.towns.copy(),
            'earning_power': self.earning_power.copy(),# potentially obsolete parameter!
            'longest_street_for_this_player': self.longest_street_for_this_player,
            'free_nodes_on_board': self.free_nodes_on_board.copy() if self.free_nodes_on_board is not None else None,
            'free_edges_on_board': self.free_edges_on_board.copy() if self.free_edges_on_board is not None else None,
        }
        return status
    
    def from_player_status(self, status: dict = dict([])) -> None:
        """
        Recreates player from status dict (generated by player.create_player_status().
        Function populates existing instance, does not return a new instance.

        Args:
            status (dict, optional): saved status. Defaults to dict([]).
        """
        
        self.name = status['name']
        self.preference = status['preference']
        self.hand = status['hand'].copy()
        self.streets = status['streets'].copy()
        self.villages = status['villages'].copy()
        self.towns = status['towns'].copy()
        self.earning_power = status['earning_power'].copy() # potentially obsolete parameter!
        self.longest_street_for_this_player = status['longest_street_for_this_player']
        self.free_nodes_on_board = status['free_nodes_on_board'].copy() if status['free_nodes_on_board'] is not None else None
        self.free_edges_on_board = status['free_edges_on_board'].copy() if status['free_edges_on_board'] is not None else None
        return 
    


    def create_vector_for_player(self, position: int, include_values: bool = True) -> np.ndarray[np.float32]:
        '''
        Create a vector representation of the player for use in machine learning models.
        The vector is created based on the player's hand, streets, villages, towns, and other attributes.
        
        Default the vector for nodes and edges is created for a player at 'position 0' on the board,
        so edges with a street are set to 1, and nodes with a village are set to 1 and nodes with a town are set to 5.

        If position is not 0, the vector is created for the player at that position.

        The layout of the vector is defined by player.structure (the BoardStructure attribute used in the initialization).

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
    
    def generate_possible_actions_for_trading_with_other_player(self,                                                        
                                                                rejected_trades: set = dict([]),
                                                                card_out_in: tuple = None
                                                                ) -> list:
        """
        Generate a list of possible actions to trade with other players, without specifying which player.
        So, this is equivalent to a general call "does someone wants to trade resource A for resource B"

        A dict of already rejected trades is passed to avoid making the same proposal twice.

        Args:
            rejected_trades (set, optional): Defaults to dict([]).
            card_out_in (tuple, optional): Defaults to None.

        Returns:
            list: list of possible action.
        """
        
        if not card_out_in:
            swaps_to_explore =   [
                (card_out,card_in) for card_out in range(len(self.hand))
                    for card_in in range(len(self.hand)) if 
                    (card_out != card_in and self.hand[card_out] > 0 and (card_out,card_in) not in rejected_trades)
                    ]
        else:
            swaps_to_explore = [(card_out_in[0],card_out_in[1])]
        results = []
        for card_out,card_in in swaps_to_explore:
            action = ('trade_player',(card_out,card_in))
            results.append(action)
        return results
    
    def generate_possible_actions_for_trading_with_specific_player(self,
                                                                rejected_trades_for_specific_player: set = dict([]),
                                                                card_out_in_partner: tuple = None
                                                                ) -> list:
        """
        Generate possible actions for trading with a specific player.
        This is a placeholder method and should be implemented in subclasses.

        A dict of already rejected trades is passed to avoid making the same proposal twice.

        """
        results = []
        # This method should be overridden to implement specific trading logic]
        # results should be a list of tuples [((card_out, card_in), trading_partner), ...]
        return results

    def generate_possible_actions_for_trading_with_bank(self,
                                                        card_out_in: tuple = None
                                                        ) -> list:
        """
        Generate a list of actions containing all possible trades for this player with the bank.

        Args:
            card_out_in (tuple, optional): Defaults to None.

        Returns:
            list: list of possible actions
        """
        if not card_out_in:
            swaps_to_explore =   [(card_out,card_in) for card_out in range(len(self.hand))
                    for card_in in range(len(self.hand)) if (card_out != card_in and self.hand[card_out] >= 4)]
        else:
            swaps_to_explore = [(card_out_in[0],card_out_in[1])]
        results = []
        for card_out,card_in in swaps_to_explore:
            action = ('trade_bank',(card_out,card_in))
            results.append(action)
        return results

    def generate_possible_actions_for_building_street(self,
                                                      edge: tuple = None, 
                                                      set_up: bool = False, 
                                                      set_up_node: int = None
                                                      ) -> list:
        """
        Generate all options to build a street for this player.
        Function does not check whether player has resource cards, just indicates
        where a street could be build.

        Default function checks all edges that link with existing streets and are not yet occupied. If we
        use argument 'edge' the function will only check that edge and return the action if it is possible.
        If we pass 'set_up = True' the function considers the set-up phase on the game, and only look for streets
        that connect to the node specified in 'set_up_node'.

        Args:
            edge (tuple, optional): If we pass a single edge the function will only check this edge. Defaults to None.
            set_up (bool, optional): True if this is the set-up phase, False during regular gameplay. Defaults to False.
            set_up_node (int, optional): Node to connect to if set_up is True. Defaults to None.

        Returns:
            list: list of possible actions.
        """
        if not set_up:
            if not edge:
                edges_to_explore = np.nonzero(self.build_options['street_options'])[0]
            else:
                edges_to_explore = [edge]
        else:
            edges_to_explore = [edge for edge,connecting_nodes in enumerate(self.structure.nodes_connected_by_edge) if set_up_node in connecting_nodes]
        results = []
        for edge in edges_to_explore:
            action = ('street',edge)
            results.append(action)
        return results
    
    def generate_possible_actions_for_building_village(self, 
                                                       node: int = None, 
                                                       set_up: bool = False
                                                       ) -> list:
        """
        Generate all options to build a village for this player.
        Function does not check whether player has resource cards, just indicates
        where a village could be build.

        Default function checks all nodes that are not yet occupied. If we
        use argument 'node' the function will only check that node and return the action if it is possible.
        
        If we pass 'set_up = True' the function considers the set-up phase on the game, and looks for all free nodes on the board.

        Args:
            node (tuple, optional): If we pass a single node the function will only check this node. Defaults to None.
            set_up (bool, optional): True if this is the set-up phase, False during regular gameplay. Defaults to False.

        Returns:
            list: list of possible actions
        """
        if not set_up:
            if not node:
                nodes_to_explore = np.nonzero(self.build_options['village_options'])[0]
            else:
                nodes_to_explore = [node]
        else:
            nodes_to_explore = np.nonzero(self.free_nodes_on_board)[0]
       
        results = []
        for node in nodes_to_explore:
            action = ('village',node)
            results.append(action)
        return results
    
    def generate_possible_actions_for_building_town(self, 
                                                    node:int = None
                                                    ) -> list:
        """
        Generate all options to build a town for this player.

        Args:
            node (int, optional): If we pass a single node the function will only check this node. Defaults to None.

        Returns:
            list: list of possible actions
        """
        if not node:
            nodes_to_explore = np.nonzero(self.villages)[0]
        else:
            nodes_to_explore = [node]
        results = []
        for node in nodes_to_explore:
            action = ('town',node)
            results.append(action)
        return results
    
    def generate_list_of_possible_actions(self,
                                            rejected_trades: set = dict([]),
                                            rejected_trades_for_specific_player: set = None
                                            ) -> list:
        '''
        Generate a list of all possible actions for the player.
        The actions are given as a list of tuples, where the first element is the action type
        and the second element is the action parameters.
        The action types are:
        - 'street': build a street on the given edge 
        - 'village': build a village on the given node
        - 'town': build a town on the given node
        - 'trade_player': trade with another player
        - 'trade_specific_player': trade with a specific player
        - 'trade_bank': trade with the bank

        NOTE: The action (None,None) is NOT added to the list. The player always has the option to pass to the next player, 
        but this is not considered an action in the return list of this function. If not action other than passing is possible,
        the function will return an empty list.
        '''
        possible_actions = []
        if self.can_build_street():
            possible_actions += self.generate_possible_actions_for_building_street()
        if self.can_build_village():
            possible_actions += self.generate_possible_actions_for_building_village()
        if self.can_build_town():
            possible_actions += self.generate_possible_actions_for_building_town()
        if self.can_trade_with_player():
            possible_actions += self.generate_possible_actions_for_trading_with_other_player(rejected_trades=rejected_trades)
        if self.can_trade_with_bank():
            possible_actions += self.generate_possible_actions_for_trading_with_bank()
        if self.can_trade_with_specific_player():
            possible_actions += self.generate_possible_actions_for_trading_with_specific_player(
                rejected_trades_for_specific_player=rejected_trades_for_specific_player
                )
        return possible_actions
    
 
    def generate_values_for_possible_actions(self, list_of_options):
        """
        Generate values for a list of options. The values are returned as a list of lists, where each inner list
        contains the values for each player in the game after executing the action.

        NOTE: For the action trade player the trade is enforced with next player, so the generated value only exactly matches
        if the next player accepts the trade.

        [
        [value player 1, value player 2, ...] # for first action in list of options,
        [value player 1, value player 2, ...] # for second action in list of options,
        ...
        ]

        The action types are:
        - 'street': build a street on the given edge
        - 'village': build a village on the given node
        - 'town': build a town on the given node
        - 'trade_player': trade with another player
        - 'trade_specific_player': trade with a specific player
        - 'trade_bank': trade with the bank

        Args:
            list_of_options (tuple): list of actions to evaluate, e.g. [('street',1),('village',2),('trade_player',(1,0))]

        Returns:
            list: A list of lists containing the values for each player in the game after executing the action.
        """
        self._player_position = self._players_in_this_game.index(self)
        values = []
        for action in list_of_options:
            
            new_board = self._board.recreate()
            #new_player = self.copy()
            #new_board.players[self._player_position] = new_player
            new_board.players = [p.copy() for p in self._players_in_this_game]
            new_player = new_board.players[self._player_position]
            new_board.inform_players_of_the_board_and_position()
            new_board.sync_status_between_board_and_players()
            new_board.execute_player_action(new_player, action, enforce_trade=True)
            values.append([p.calculate_value() for p in new_board.players])
        return np.array(values)
    
    def generate_values_for_options(self, list_of_options):
        """
        Refers to generate_values_for_possible_actions
        """
        warnings.warn("This function is deprecated, use generate_values_for_possible_actions() instead", DeprecationWarning)
        return self.generate_values_for_possible_actions(list_of_options)

    def random_action(self,
                         rejected_trades_for_this_round: set = set([]),
                         rejected_trades_for_this_round_for_specific_player: set = set([])
                         ) -> tuple:
        """
        Return random action from list of possible actions PLUS the option to pass to the next player.
        If no other actions are possible function will always return (None,None) to indicate that the player
        passes to the next player.

        Args:
            rejected_trades_for_this_round (set, optional). Defaults to set([]).
            rejected_trades_for_this_round_for_specific_player (set, optional). Defaults to set([]).

        Returns:
            tuple: action, e.g. ('street',1)
        """
        # === generate list of all options ===
        possible_actions = self.generate_list_of_possible_actions(
                                        rejected_trades=rejected_trades_for_this_round,
                                        rejected_trades_for_specific_player=rejected_trades_for_this_round_for_specific_player
                                        )
        # === add action to pass to next player ===
        possible_actions.append((None,None))
        random_index = random.randint(a=0,b=len(possible_actions)-1) # random.randint includes a and b
        return possible_actions[random_index]
    
    def random_build_action(self) -> tuple:
        """
        Return random build action from list of possible actions (i.e, building a street, village or town).
        If there are no possible build actions, return (None,None), otherwise return a random build action (so
        if there are options the function will never return (None,None)).

        Args:
            rejected_trades_for_this_round (set, optional). Defaults to set([]).
            rejected_trades_for_this_round_for_specific_player (set, optional). Defaults to set([]).

        Returns:
            tuple: action, e.g. ('street',1)
        """
        # === generate list of all options ===
        possible_actions = self.generate_list_of_possible_build_actions(
                                        rejected_trades=dict([]),
                                        rejected_trades_for_specific_player=dict([])
                                        )
        possible_build_actions = [action for action in possible_actions if action[0] in ['street','village','town']]
        if len(possible_build_actions) == 0:
            return (None,None)
        random_index = random.randint(a=0,b=len(possible_actions)-1) # random.randint includes a and b
        return possible_actions[random_index]

    def random_trade_action(self,
                            rejected_trades_for_this_round: set = set([]),
                            rejected_trades_for_this_round_for_specific_player: set = set([])
                            ) -> tuple:
        """
        Return random trade action from list of possible actions (i.e, trading resources).
        If there are no possible trade actions, return (None,None), otherwise return a random trade action (so
        if there are options the function will never return (None,None)).

        Args:
            rejected_trades_for_this_round (set, optional). Defaults to set([]).
            rejected_trades_for_this_round_for_specific_player (set, optional). Defaults to set([]).

        Returns:
            tuple: action, e.g. ('trade_player', (1,0))
        """
        # === generate list of all options ===
        possible_actions = self.generate_list_of_possible_actions(
                                        rejected_trades=rejected_trades_for_this_round,
                                        rejected_trades_for_specific_player=rejected_trades_for_this_round_for_specific_player
                                        )
        possible_trade_actions = [action for action in possible_actions if action[0] in ['trade_player','trade_bank','trade_specific_player']]
        if len(possible_trade_actions) == 0:
            return (None,None)
        random_index = random.randint(a=0,b=len(possible_trade_actions)-1) # random.randint includes a and b
        return possible_trade_actions[random_index]

    
    
    

    
 