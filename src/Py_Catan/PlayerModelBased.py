from Py_Catan.Preferences import PlayerPreferences
from Py_Catan.BoardStructure import BoardStructure
from Py_Catan.Player import Player
from Py_Catan.BoardVector import BoardVector
import numpy as np
from keras.models import Model

class Player_Model_Based(Player):
    def __init__(self,
                 name: str = 'A', 
                 structure: BoardStructure = BoardStructure(),
                 model: any = Model(), 
                 status: dict = dict([]) ):
        # Call to Player's constructor
        super().__init__(
                name = name, 
                structure = structure,
                status = status
                )
        self.model = model
        self._player_position = None
        self._board = None


    def copy(self) -> 'Player_Model_Based':
        '''
        Create a copy of the player with the same attributes.
        The name is set to 'New' to avoid conflicts with the original player.
        '''
        #new_player = Player_Model_Based(name ='New',structure=self.structure, preference=self.preference)
        new_player = Player_Model_Based(name ='New')
        np = vars(new_player)
        for k,v in vars(self).items():
            if k != 'name':
                try:
                    np[k] = v.copy()
                except:
                    np[k] = v
        return new_player

    def respond_positive_to_other_players_trading_request(self,card_out_in) -> bool:
        '''        
        Return 'True' if this player responds positively to a trading request from another player.
        The player will accept the trade if they have at least one card of the type they are receiving
        and at least one card of the type they are giving away.
        The player will also check if the trade is beneficial for them.
        If the trade is beneficial, the player will accept it.  
        '''
        if self.hand[card_out_in[0]] <= 0:
            return False
        current_value = self.calculate_value((None,None))
        new_value = self.calculate_value(('trade_player', card_out_in))
        return (new_value > current_value) and not np.isclose(new_value,current_value, atol=self.atol)
   
    # def generate_possible_actions_for_trading_with_other_player(self,                                                        
    #                                                             rejected_trades: set = None,
    #                                                             card_out_in: tuple = None
    #                                                             ) -> list:
        
    #     if not card_out_in:
    #         swaps_to_explore =   [
    #             (card_out,card_in) for card_out in range(len(self.hand))
    #                 for card_in in range(len(self.hand)) if 
    #                 (card_out != card_in and self.hand[card_out] > 0 and (card_out,card_in) not in rejected_trades)
    #                 ]
    #     else:
    #         swaps_to_explore = [(card_out_in[0],card_out_in[1])]
    #     results = []
    #     for card_out,card_in in swaps_to_explore:
    #         action = ('trade_player',(card_out,card_in))
    #         results.append(action)
    #     return results
    
    # def generate_possible_actions_for_trading_with_specific_player(self,
    #                                                             rejected_trades_for_specific_player: set = None,
    #                                                             card_out_in: tuple = None
    #                                                             ) -> list:
    #     """
    #     Generate possible actions for trading with a specific player.
    #     This is a placeholder method and should be implemented in subclasses.
    #     """
    #     results = []
    #     # This method should be overridden to implement specific trading logic]
    #     # results should be a list of tuples [((card_out, card_in), trading_partner), ...]
    #     return results

    # def generate_possible_actions_for_trading_with_bank(self,
    #                                                     card_out_in: tuple = None
    #                                                     ) -> list:
    #     if not card_out_in:
    #         swaps_to_explore =   [(card_out,card_in) for card_out in range(len(self.hand))
    #                 for card_in in range(len(self.hand)) if (card_out != card_in and self.hand[card_out] >= 4)]
    #     else:
    #         swaps_to_explore = [(card_out_in[0],card_out_in[1])]
    #     results = []
    #     for card_out,card_in in swaps_to_explore:
    #         action = ('trade_bank',(card_out,card_in))
    #         results.append(action)
    #     return results

    # def generate_possible_actions_for_building_street(self,
    #                                                   edge: tuple = None, 
    #                                                   set_up: bool = False, 
    #                                                   set_up_node: int = None
    #                                                   ) -> list:
    #     if not set_up:
    #         if not edge:
    #             edges_to_explore = np.nonzero(self.build_options['street_options'])[0]
    #         else:
    #             edges_to_explore = [edge]
    #     else:
    #         edges_to_explore = [edge for edge,connecting_nodes in enumerate(self.structure.nodes_connected_by_edge) if set_up_node in connecting_nodes]
    #     results = []
    #     for edge in edges_to_explore:
    #         action = ('street',edge)
    #         results.append(action)
    #     return results
    
    # def generate_possible_actions_for_building_village(self, 
    #                                                    node: int = None, 
    #                                                    set_up: bool = False
    #                                                    ) -> list:
    #     if not set_up:
    #         if not node:
    #             nodes_to_explore = np.nonzero(self.build_options['village_options'])[0]
    #         else:
    #             nodes_to_explore = [node]
    #     else:
    #         nodes_to_explore = np.nonzero(self.free_nodes_on_board)[0]
       
    #     results = []
    #     for node in nodes_to_explore:
    #         action = ('village',node)
    #         results.append(action)
    #     return results
    
    # def generate_possible_actions_for_building_town(self, 
    #                                                 node:int = None
    #                                                 ) -> list:
    #     if not node:
    #         nodes_to_explore = np.nonzero(self.villages)[0]
    #     else:
    #         nodes_to_explore = [node]
    #     results = []
    #     for node in nodes_to_explore:
    #         action = ('town',node)
    #         results.append(action)
    #     return results
    
    # def generate_list_of_possible_actions(self,
    #                                         rejected_trades: set = None,
    #                                         rejected_trades_for_specific_player: set = None
    #                                         ) -> list:
    #     '''
    #     Generate a list of all possible actions for the player.
    #     The actions are given as a list of tuples, where the first element is the action type
    #     and the second element is the action parameters.
    #     The action types are:
    #     - 'street': build a street on the given edge 
    #     - 'village': build a village on the given node
    #     - 'town': build a town on the given node
    #     - 'trade_player': trade with another player
    #     - 'trade_specific_player': trade with a specific player
    #     - 'trade_bank': trade with the bank
    #     '''
    #     possible_actions = []
    #     if self.can_build_street():
    #         possible_actions += self.generate_possible_actions_for_building_street()
    #     if self.can_build_village():
    #         possible_actions += self.generate_possible_actions_for_building_village()
    #     if self.can_build_town():
    #         possible_actions += self.generate_possible_actions_for_building_town()
    #     if self.can_trade_with_player():
    #         possible_actions += self.generate_possible_actions_for_trading_with_other_player(rejected_trades=rejected_trades)
    #     if self.can_trade_with_bank():
    #         possible_actions += self.generate_possible_actions_for_trading_with_bank()
    #     if self.can_trade_with_specific_player():
    #         possible_actions += self.generate_possible_actions_for_trading_with_specific_player(
    #             rejected_trades_for_specific_player=rejected_trades_for_specific_player
    #             )
    #     return possible_actions

    def generate_values_for_possible_actions(self,
                                             possible_actions: list
                                             ) -> list:
        '''
        Generate values for all possible actions.
        The actions are given as a list of tuples, where the first element is the action type
        and the second element is the action parameters.
        The action types are:
        - 'street': build a street on the given edge
        - 'village': build a village on the given node
        - 'town': build a town on the given node
        - 'trade_player': trade with another player
        - 'trade_specific_player': trade with a specific player
        - 'trade_bank': trade with the bank
        '''
        if not possible_actions:
            raise Exception('No possible actions to evaluate')
        initial_vector = BoardVector(board=self._board, include_values = False)
        x1,x2,x3 = [],[],[]
        for action in possible_actions:
            if action[0] == None:
                new_vector = initial_vector.vector.copy()
            elif action[0] == 'street':
                new_vector = initial_vector.build_street(self._player_position,action[1]  )
            elif action[0] == 'village':
                new_vector = initial_vector.build_village(self._player_position,action[1]  )
            elif action[0] == 'town':
                new_vector = initial_vector.build_town(self._player_position,action[1]  )
            elif action[0] == 'trade_player':
                new_vector = initial_vector.trade_between_players(self._player_position,action[1]  )
            elif action[0] == 'trade_specific_player':
                new_vector = initial_vector.trade_between_players(self._player_position,action[1][0], action[1][1])
            elif action[0] == 'trade_bank':
                new_vector = initial_vector.trade_with_bank(self._player_position,action[1]  )
            else:
                raise Exception(f'Unknown action: {action[0]}')
            
            vector_components = initial_vector.split_vector(vector=new_vector)
            input1 = vector_components[-3].astype(np.int32)
            x1.append(input1)
            input2 = vector_components[-2].astype(np.int32)
            x2.append(input2)
            input3 = vector_components[-1].astype(np.int32)
            x3.append(input3)
        values = self.model.predict([np.array(x1,dtype=np.int32), np.array(x2,dtype=np.int32), np.array(x3,dtype=np.int32)], verbose=0)
        return values
    
    def find_best_action(self, 
                         rejected_trades_for_this_round: set = set([]),
                         rejected_trades_for_this_round_for_specific_player: set = set([])
                         ) -> tuple:
        '''
        Find the best action for the player based on the current state of the board.
        The action is given as a tuple, where the first element is the action type
        and the second element is the target (e.g., node or edge) for the action.
        
        If no action is better than the current state, return (None, None).
        '''
        initial_value = self.calculate_value()
        possible_actions = self.generate_list_of_possible_actions(rejected_trades=rejected_trades_for_this_round)
        if possible_actions:
            values = self.generate_values_for_possible_actions(possible_actions)[..., self._player_position]
            best_index = np.argmax(values)
        if not possible_actions or values[best_index] <= initial_value:
            best_action = (None, None)  # No action is better than the current state
        else:
            best_action = possible_actions[best_index]
        return best_action
    
    
    def player_setup(self,brd) -> None:
        '''
        Setup the player with initial buildings.

        Returns a list of two actions to perform during setup 
        (one village and one street).
        '''
        self.update_build_options()
        self.hand = np.array(brd.structure.real_estate_cost[1])
        options = self.generate_possible_actions_for_building_village(set_up=True)
        if not options:
            raise Exception('No options for building village during setup')
        values = self.generate_values_for_possible_actions(options)[..., self._player_position]
        best_index = np.argmax(values)
        best_node = options[best_index][1]
        self.update_build_options()
        self.hand += np.array(brd.structure.real_estate_cost[0])
        options = self.generate_possible_actions_for_building_street(set_up=True, set_up_node=best_node)
        if not options:
            raise Exception('No options for building street during setup')
        values = self.generate_values_for_possible_actions(options)[..., self._player_position]
        best_index = np.argmax(values)
        best_edge = options[best_index][1]
        return [('village', best_node),('street',best_edge)]
    
    def calculate_value(self, action: tuple = (None,None)):
        '''
        Calculate the value of the player after performing the given action.
        The action is given as a tuple, where the first element is the action type
        and the second element is the target (e.g., node or edge) for the action.
        
        Value is returned as a float.
        '''
        initial_vector = BoardVector(board=self._board, include_values = False)
        nodes = initial_vector.indices['nodes']
        edges = initial_vector.indices['edges']
        hands = initial_vector.indices['hands']
        
        if action[0] == None:
            new_vector = initial_vector.vector.copy()
        elif action[0] == 'street':
            new_vector = initial_vector.build_street(self._player_position,action[1]  )
        elif action[0] == 'village':
            new_vector = initial_vector.build_village(self._player_position,action[1]  )
        elif action[0] == 'town':
            new_vector = initial_vector.build_town(self._player_position,action[1]  )
        elif action[0] == 'trade_player':
            new_vector = initial_vector.trade_between_players(self._player_position,action[1]  )
        elif action[0] == 'trade_bank':
            new_vector = initial_vector.trade_with_bank(self._player_position,action[1]  )
        else:
            raise Exception(f'Unknown action: {action[0]}')
        input1 = np.array([new_vector[nodes]], dtype=np.int32)
        input2 = np.array([new_vector[edges]], dtype=np.int32)
        input3 = np.array([new_vector[hands]], dtype=np.int32)
        values = self.model.predict([input1, input2, input3], verbose=0)
        return values[0][self._player_position]
    
    
  
 