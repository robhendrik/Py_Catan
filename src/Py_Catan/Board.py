import sys  
import numpy as np
sys.path.append("./src")
from Py_Catan.BoardStructure import BoardStructure, BoardLayout
from Py_Catan.Preferences import PlayerPreferences
from Py_Catan.Player import Player

class Board:
    def __init__(self,
                 structure: BoardStructure = BoardStructure(),
                 players: list = [],
                 status: dict = dict([])):
        if status:
            self.structure = status['structure']
            self.occupied_nodes = status['occupied_nodes'].copy()
            self.occupied_edges = status['occupied_edges'].copy()
            self.players = [Player(status = player_status) for player_status in status['players']]
        elif structure and players:
            self.structure = structure
            self.occupied_nodes = np.zeros(self.structure.no_of_nodes)
            self.occupied_edges = np.zeros(self.structure.no_of_edges)
            self.players = players
        elif structure:
            self.structure = structure
            self.occupied_nodes = np.zeros(self.structure.no_of_nodes)
            self.occupied_edges = np.zeros(self.structure.no_of_edges)
            self.players = [Player(name = name,
                                preference = PlayerPreferences(),
                                structure = self.structure)
                                for name in ['A', 'B', 'C', 'D']]
        else:
            raise Exception("Either structure or status must be provided to initialize the board.")
        
        self.free_nodes_on_board = self.free_nodes()
        self.free_edges_on_board = self.free_edges()
        self._print_player_action = False  # Set to True to print player actions for debugging

    def recreate(self) -> 'Board':
        '''
        Recreate the board as a new instance with the same structure and players position.
        The players are of the base class.
        This is useful for creating a copy of the board without modifying the original instance.
        '''
        new_board = Board(structure=self.structure)
        new_board.occupied_nodes = self.occupied_nodes.copy()
        new_board.occupied_edges = self.occupied_edges.copy()
        new_board.players = [Player(name=p.name, preference=p.preference, structure=self.structure) for p in self.players]
        for new_p, old_p in zip(new_board.players, self.players):
            new_p.hand = old_p.hand.copy()
            new_p.villages = old_p.villages.copy()
            new_p.towns = old_p.towns.copy()
            new_p.streets = old_p.streets.copy()
            new_p.longest_street_for_this_player = old_p.longest_street_for_this_player
            new_p.owns_longest_street = old_p.owns_longest_street
            new_p._board = new_board
            new_p._players_in_this_game = new_board.players
        return new_board

    def free_nodes(self) -> np.ndarray  :
        '''
        Return the free nodes on the board.
        A node is free if it does not have a village or town on it,
        or on its direct neighbout.
        '''
        return np.logical_not(self.occupied_nodes  @ self.structure.node_neighbour_matrix)
    
    def free_edges(self) -> np.ndarray:
        '''
        Return the free edges on the board.
        An edge is free if it does not have a street on it.
        '''
        return np.logical_not(self.occupied_edges)

    def build_street(self,player,edge) ->  None:
        '''
        Build a street on the board for player.
        Update longest street for that player.
        Update occupied edges on the board.
        Update the board for all players, including determining who owns the longest street.
        '''
        
        player._longest_street_has_been_updated = False  # Reset the longest street update flag
        player.build_street(edge)
        if not player._longest_street_has_been_updated:
            player.calculate_longest_street()   # Calculate longest street if not already done
        self.occupied_edges[edge] = 1
        # _update_board_for_players takes player.longest_street_for_this_player into account, 
        # but does not recalculate it. We recalculate in the player instance when calling player.build_street.
        self._update_board_for_players()
        # Updating longest street across all players is done in _update_board_for_players.
        return 

    def build_village(self,player,node) -> None:
        ''' 
        Build a village on the board.
        Update occupied nodes on the board
        The node is marked as occupied and the player's village count is updated.
        The player's hand is updated with the resources used to build the village.
        The free nodes on the board are updated.    
        '''
        player.build_village(node)
        self.occupied_nodes[node] = 1
        self._update_board_for_players()
        return 

    def build_town(self,player,node) -> None:
        '''
        Build a town on the board.
        The node must be occupied by a village of the player.       
        The player's hand is updated with the resources used to build the town.
        '''
        player.build_town(node)
        # self._update_board_for_players() should not make a difference since town location is already occupied by a village
        return 
    
    def trade_with_bank(self, player, card_out_in) -> bool:
        '''
        Trade with the bank.
        The player gives out 4 of card_out_in[0] and receives 1 of card_out_in[1].
        The trade is executed if the player has enough resources.
        Returns True if the trade was successful, False otherwise.
        '''
        if player.hand[card_out_in[0]] >= 4:
            player.trade_with_bank(card_out_in=(card_out_in[0],card_out_in[1]))
            return True
        return False

    
    def propose_and_execute_trade(self, player, card_out_in, specified_trading_partner: int = None) -> bool:
        ''' 
        Propose a trade to other players.
        The player proposes to give out card_out_in[0] and receive card_out_in[1].
        The other players can respond to the trade.
        If a player accepts the trade, the trade is executed (and function returns True).
        If no player accepts the trade, the trade is not executed (and function returns False).

        If specified_trading_partner is not None, the trade is only proposed to that player.
        If specified_trading_partner is None, the trade is proposed to all players.

        returns:
            bool: True if the trade was accepted and executed, False otherwise.
        '''
        if specified_trading_partner is not None:
            new_player_order = [self.players[specified_trading_partner]]
        else:
            shift = self.players.index(player)
            new_player_order = np.roll(self.players, shift = -1*shift)
        for q in new_player_order:
            if q == player or q.hand[card_out_in[1]] <= 0:
                continue
            answer = q.respond_positive_to_other_players_trading_request(card_out_in=(card_out_in[1],card_out_in[0]))
            if answer == True:
                player.trade_with_player(card_out_in=(card_out_in[0],card_out_in[1]))
                q.trade_with_player(card_out_in=(card_out_in[1],card_out_in[0]))
                return True
        return False
    
    def enforce_and_execute_trade(self, player, card_out_in, specified_trading_partner: int = None) -> None:
        ''' 
        Trade is always executed either with next player or with specified trading partner. Potentially
        partner ends up with negative resources! This is for simulation purposes only.
        '''
        if specified_trading_partner is not None:
            new_player_order = [self.players[specified_trading_partner]]
        else:
            shift = self.players.index(player)
            new_player_order = np.roll(self.players, shift = -1*shift)
        enforced_trading_partner = new_player_order[1]
        
       
        player.trade_with_player(card_out_in=(card_out_in[0],card_out_in[1]))
        enforced_trading_partner.trade_with_player(card_out_in=(card_out_in[1],card_out_in[0]))
        return
    
    


    def throw_dice(self, enforce: int = -1) -> int:
        '''
        Throw the dice and return the value.
        If enforce is -1, the dice are thrown randomly. 
        If enforce is a number, the dice are set to that value.
        If the dice value is in self.values, the players receive resources.
        If the dice value is equal to self.dice_value_to_hand_in_cards,
        the players have to hand in half of their resources.
        '''
        if enforce == -1:
            dice_1 = np.random.choice([1,2,3,4,5,6])
            dice_2 = np.random.choice([1,2,3,4,5,6])
        else:
            dice_1,dice_2 = enforce, 0

        if (dice_1+dice_2) in self.structure.values:
            dice_value = self.structure.dice_results.index(dice_1 + dice_2)
            for player in self.players:
                player.hand += (player.villages + 2*player.towns)@self.structure.dice_impact_per_node_dnt[dice_value]
                player.hand = player.hand.copy()
        # when you throw 7 you have to hand in half of your resources
        elif (dice_1+dice_2) == self.structure.dice_value_to_hand_in_cards:
            for player in self.players:
                if sum(player.hand) > 7:
                    qty_to_remove = sum(player.hand) // 2
                    for _ in range(qty_to_remove):
                        options = np.nonzero(player.hand)[0]
                        card_out = np.random.choice(options)
                        player.hand[card_out] -= 1
        for player in self.players:
            player._actions_in_round = 0
        return (dice_1+dice_2)
    
    def create_board_status(self)-> dict:
        '''Create a status of the board that can be used to save the game.'''
        board_status = {
            'structure': self.structure,
            'occupied_nodes': self.occupied_nodes.copy(),
            'occupied_edges': self.occupied_edges.copy(),
            'players': [p.create_player_status() for p in self.players]
        }
        return board_status

    def create_board_vector(self, include_values: bool = True) -> np.array:
        '''
        Create a vector representation of the board.
        The vector contains the occupied nodes and edges, and the players' hands.
        The vector is used for training the neural network.
        '''
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
        # Update the board for players before creating the vector
        # This ensures that the players' free edges and nodes are up to date.
        # Especially for value function and models this can be important.
        self._update_board_for_players()
        for p in self.players:
            p.update_build_options()
        # We can add the vectors for players, since they cannot occupy the same node or edges and have hands and values in different indices.
        # If we ever include game related entries, we have to be careful in teh addition.
        vector = np.zeros(len(self.structure.header), np.float32)
        for position,player in enumerate(self.players):
            vector += player.create_vector_for_player(position=position, 
                                                       include_values=include_values)

        return vector

    def logging_header(self):
        '''
        Create a header for logging the board state.
        Refers to the logging header of the structure.
        '''
        return self.structure.header


    def _update_board_for_players(self) -> None:
        '''
        Update the board for all players based on their current state.
        This method updates the free edges and nodes on the board for each player, 
        as well as the longest street on the board.
        '''
        self.free_nodes_on_board = self.free_nodes()
        self.free_edges_on_board = self.free_edges()
        longest_streets = [p.longest_street_for_this_player for p in self.players]
        # m is the longest street length, but never smaller than 3 (the start value)
        m = max(max(longest_streets),self.structure.longest_street_minimum)
        for p in self.players:
            p.free_edges_on_board = self.free_edges_on_board
            p.free_nodes_on_board = self.free_nodes_on_board
            p.longest_street_on_board = m
            if p.longest_street_for_this_player == m and longest_streets.count(m) == 1:
                p.owns_longest_street = True
            else:
                p.owns_longest_street = False
            # === We ask to call inform_players_of_the_board_and_position() in the game class,
            # but we can also call it here to ensure that the players have the correct board and position.
            p._board = self
            p._players_in_this_game = self.players
            p._player_position = self.players.index(p)
            # ----------------------
        return
    
    def inform_players_of_the_board_and_position(self):
        for p in self.players:
            p._board = self
            p._players_in_this_game = self.players
            p._player_position = self.players.index(p)
        return

    def sync_status_between_board_and_players(self):
        """
        Update build options for all players. 
        Executes board._update_board_for_players() and player.update_build_options() for each player.
        
        After this function:
        - Fee nodes and edges on the board are updated (recalculated from occupied nodes and edges).
        - Each player has their free nodes and edges updated by copying value from the board.
        - Each player has their build options updated based on the current state of the board.
        - The longest street on the board is determined and set (with minimum from structure) in player.longest_street_on_board.
        - The flag 'player.owns_longest_street' is set to True if the player has the longest street and larger than minimum,
          otherwise it is set to False.
        - Also includes the function in inform_players_of_the_board_and_position() to ensure that players have the correct board and position.
        """
        self._update_board_for_players()
        for p in self.players:
            p.update_build_options()
        return
    
    def execute_player_action(self,
                               player, 
                               best_action: tuple = (None,None), 
                               rejected_trades_for_this_round: set = set([]),
                               rejected_trades_for_this_round_for_specific_player: set = set([]),
                               enforce_trade: bool = False
                               ) -> None:
        '''
        Execute the best action for the player.
        The action is a tuple of the form (action_type, action_value).
        The action_type can be one of the following:
            - 'street': build a street on the board
            - 'village': build a village on the board
            - 'town': build a town on the board
            - 'trade_player': trade with another player
            - 'trade_specific_player': trade with a specific player
            - 'trade_bank': trade with the bank
        The action_value is the edge or node where the action is performed, or the card_out_in tuple for trading.
        If the action is not valid, it is ignored.
        The method updates the board and the player state accordingly.
        '''
        if self._print_player_action == True:
            print(f"Player {player.name} executing action: {best_action}")
        if best_action[0] == 'street':
            self.build_street(player=player,edge=best_action[1])
        elif best_action[0] == 'village':
            self.build_village(player=player,node=best_action[1])
        elif best_action[0] == 'town':
            self.build_town(player=player,node = best_action[1])
        elif best_action[0] == 'trade_player':
            if not enforce_trade:
                response = self.propose_and_execute_trade(player = player, card_out_in = best_action[1])
                if response == False:
                    rejected_trades_for_this_round.add(best_action[1])
                    if self._print_player_action == True:
                        print(f"Trade {best_action[1]} rejected for player {player.name}.")
            else:
                self.enforce_and_execute_trade(player = player, card_out_in = best_action[1])
   
        elif best_action[0] == 'trade_specific_player':
            if not enforce_trade:
                response = self.propose_and_execute_trade(player = player, 
                                            card_out_in = best_action[1][0], 
                                            specified_trading_partner = best_action[1][1])
                if response == False:
                    rejected_trades_for_this_round_for_specific_player.add(best_action[1])
                    if self._print_player_action == True:
                        print(f"Trade {best_action[1]} rejected for player {player.name}.")
            else:
                self.enforce_and_execute_trade(player = player, 
                                            card_out_in = best_action[1][0], 
                                            specified_trading_partner = best_action[1][1])
        elif best_action[0] == 'trade_bank':
            self.trade_with_bank(player = player, card_out_in = best_action[1])
        elif best_action == (None, None):
            pass
        else:
            raise ValueError(f"Unknown action type: {best_action[0]}")
        
        self._update_board_for_players()
        for p in self.players:
            p.update_build_options()
        return