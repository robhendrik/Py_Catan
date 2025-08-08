from Py_Catan.Preferences import PlayerPreferences
from Py_Catan.BoardStructure import BoardStructure
from Py_Catan.Player import Player
from Py_Catan.ValueFunction import ValueFunction
import numpy as np
import warnings

class Player_Value_Function_Based(Player):
    def __init__(self,
                 name: str = 'A', 
                 structure: BoardStructure = BoardStructure(),
                 preference: any = PlayerPreferences(), 
                 status: dict = dict([]) ):
        # Call to Player's constructor
        super().__init__(
                name = name, 
                structure = structure,
                #preference = preference, 
                status = status,           
                )
        self.value_function = ValueFunction(preference, structure)
        self.preference = preference
        return 
    
    def copy(self) -> 'Player_Value_Function_Based':
        '''
        Create a copy of the player with the same attributes.
        The name is set to 'New' to avoid conflicts with the original player.

        Overwrites the function in parent class.
        '''        
        new_player = Player_Value_Function_Based(name ='New', structure=self.structure, preference=self.preference)
        np = vars(new_player)
        for k,v in vars(self).items():
            if k not in ['name', 'value_function']:
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
        current_value = self.calculate_value()
        action = ('trade_player', card_out_in)
        new_value = self.generate_values_for_possible_actions([action])[..., self._player_position]
        return (new_value > current_value) and not np.isclose(new_value,current_value, atol=self.atol)
    
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
        Setup the player with initial buildings. Which streets and villages are built is based on the preferences.

        Function returns a list of actions to be executed by the board.
        The actions are tuples of the form ('street', edge) or ('village', node)
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
    
    def calculate_value(self):
        """ This function now refers to Py_Catan.ValueFunction.value_for_player()"""
        return self.value_function.value_for_player(self)
    


    
    # ============================================================================================================
    #
    # The following functions are deprecated and will be removed in the future.
    #
    # =============================================================================================================
    def player_setup_old_method(self, brd) -> list:
        """
        Setup the player with initial buildings. Which streets and villages are built is based on the preferences.

        Args:
            brd (_type_): _description_

        Returns:
            list: _description_
        """
        warnings.warn("This function is deprecated.", UserWarning)
        # self.update_build_options()
        # self.hand = np.array(brd.structure.real_estate_cost[1])
        # options = self.explore_building_village(set_up=True)
        # best_node = options[0][1]
        # self.update_build_options()
        # self.hand += np.array(brd.structure.real_estate_cost[0])
        # options = self.explore_building_street(set_up=True, set_up_node=best_node)
        # best_edge = options[0][1]
        # return [('village', best_node),('street',best_edge)]


    
    def explore_trading_with_specific_player(self,
                                              rejected_trades: set = None,
                                              rejected_trades_for_specific_player: set = None,
                                              card_out_in_partner: tuple = None) -> list:
        """
        Explore trading options with a specific player.
        Functionality not yet implemented.
        Returns a list of tuples with the action and the cards to trade, as well as the trading partner.
        NOTE: The first element in the tuple represent the value of the HAND, not total value of the player.

        Args:
            rejected_trades (set, optional): Defaults to None.
            rejected_trades_for_specific_player (set, optional): Defaults to None.
            card_out_in_partner (tuple, optional): ((card_out, card_in),trading_partner) ), Defaults to None.

        Returns:
            list[tuple]: A list of tuples with the action and the cards to trade, 
            ordered by the most optimal trade first, ( value, ((card_out, card_in),trading_partner)) 
        """
        warnings.warn("This function is not yet implemented. It will return an empty list.", UserWarning)
        results = []
        # results will be a list of tuples [( value, ((card_out, card_in),trading_partner)) ), ...]
        return results
        
    def explore_trading_with_other_player(self,
                                          rejected_trades: set = None,
                                          card_out_in: tuple = None) -> list:
        """
        Explore trading options with other players.
        Returns a list of tuples with the action and the cards to trade.
        The list is ordered with the most optimal trade first.
        NOTE: The first element in the tuple represent the value of the HAND, not total value of the player.

        Args:
            rejected_trades (set, optional): Defaults to None.
            card_out_in (tuple, optional): Defaults to None.

        Returns:
            list[tuple]: A list of tuples with the action and the cards to trade, ordered by the most optimal trade first.
        """
        warnings.warn("This function is deprecated.", UserWarning)
        # if not card_out_in:
        #     swaps_to_explore =   [
        #         (card_out,card_in) for card_out in range(len(self.hand))
        #             for card_in in range(len(self.hand)) if 
        #             (card_out != card_in and self.hand[card_out] > 0 and (card_out,card_in) not in rejected_trades)
        #             ]
        # else:
        #     swaps_to_explore = [(card_out_in[0],card_out_in[1])]
        actions_to_explore = self.generate_possible_actions_for_trading_with_other_player(
                                                rejected_trades=rejected_trades, 
                                                card_out_in=card_out_in)
        swaps_to_explore = [action[1] for action in actions_to_explore if action[0] == 'trade_player']
        results = []
        for card_out,card_in in swaps_to_explore:
            temp_player = self.copy()
            temp_player.hand[card_out] -= 1
            temp_player.hand[card_in] += 1
            results.append((temp_player.calculate_value(),  (card_out,card_in)))
            del temp_player
            # temp_hand = self.hand.copy()
            # temp_hand[card_out] -= 1
            # temp_hand[card_in] += 1
            # results.append((self.calculate_value_hand(temp_hand), (card_out,card_in)))
        results.sort(key=lambda a: a[0],reverse=True)
        return results
        
    def explore_trading_with_bank(self,card_out_in: tuple = None) -> list:
        """
        Explore trading options with the bank.
        Returns a list of tuples with the action and the cards to trade.
        The list is ordered with the most optimal trade first.
        NOTE: The first element in the tuple represent the value of the HAND, not total value of the player.

        Args:
            card_out_in (tuple, optional): Defaults to None.

        Returns:
            list[tuple]: A list of tuples with the action and the cards to trade, ordered by the most optimal trade first.
        """
        warnings.warn("This function is deprecated.", UserWarning)
        # if not card_out_in:
        #     swaps_to_explore =   [(card_out,card_in) for card_out in range(len(self.hand))
        #             for card_in in range(len(self.hand)) if (card_out != card_in and self.hand[card_out] >= 4)]
        # else:
        #     swaps_to_explore = [(card_out_in[0],card_out_in[1])]
        actions_to_explore = self.generate_possible_actions_for_trading_with_bank(card_out_in=card_out_in)
        swaps_to_explore = [action[1] for action in actions_to_explore if action[0] == 'trade_bank']
        results = []
        for card_out,card_in in swaps_to_explore:
            temp_player = self.copy()
            temp_player.hand[card_out] -= 4
            temp_player.hand[card_in] += 1
            results.append((temp_player.calculate_value(),  (card_out,card_in)))
            del temp_player
            # temp_hand = self.hand.copy()
            # temp_hand[card_out] -= 4
            # temp_hand[card_in] += 1
            # results.append((self.calculate_value_hand(temp_hand), (card_out,card_in)))
        results.sort(key=lambda a: a[0],reverse=True)
        return results

    def explore_building_street(self,
                                edge: tuple = None, 
                                set_up: bool = False, 
                                set_up_node: int = None) -> list:
        """
        Explore building options for streets.
        Returns a list of tuples with the value and the edge to build on.
        The list is ordered with the most optimal street first.

        Args:
            edge (tuple, optional): If we pass a single edge the function will only check this edge. Defaults to None.
            set_up (bool, optional): True if this is the set-up phase, False during regular gameplay. Defaults to False.
            set_up_node (int, optional): Node to connect to if set_up is True. Defaults to None.

        Returns:
            list[tuple]: A list of tuples with the value and the edge to build on.
        """
        warnings.warn("This function is deprecated.", UserWarning)
        # if not set_up:
        #     if not edge:
        #         edges_to_explore = np.nonzero(self.build_options['street_options'])[0]
        #     else:
        #         edges_to_explore = [edge]
        # else:
        #     edges_to_explore = [edge for edge,connecting_nodes in enumerate(self.structure.nodes_connected_by_edge) if set_up_node in connecting_nodes]
        actions_to_explore = self.generate_possible_actions_for_building_street(
                                                                                    edge=edge, 
                                                                                    set_up=set_up, 
                                                                                    set_up_node=set_up_node
                                                                                )
        edges_to_explore = [action[1] for action in actions_to_explore if action[0] == 'street']
        results = []
        for edge in edges_to_explore:
            temp_player = self.copy()
            temp_player.hand -= np.array(temp_player.structure.real_estate_cost[0])
            temp_player.streets[edge] = 1
            temp_player.calculate_longest_street()
            temp_player.update_build_options()
            results.append((temp_player.calculate_value(), edge))
            del temp_player
        results.sort(key=lambda a: a[0],reverse=True)
        return results
    
    def explore_building_village(self, node: int = None, set_up: bool = False):
        """
        Explore building options for villages.
        Returns a list of tuples with the value and the node to build on.
        The list is ordered with the most optimal village first.

        Args:
            node (int, optional): Defaults to None.
            set_up (bool, optional): Defaults to False.

        Returns:
            list[tuple]: A list of tuples with the value and the node to build on.
        """
        warnings.warn("This function is deprecated.", UserWarning)
        # if not set_up:
        #     if not node:
        #         nodes_to_explore = np.nonzero(self.build_options['village_options'])[0]
        #     else:
        #         nodes_to_explore = [node]
        # else:
        #     nodes_to_explore = np.nonzero(self.free_nodes_on_board)[0]
        # actions_to_explore = self.generate_possible_actions_for_building_village(
        #     node=node, 
        #     set_up=set_up
        # )
        # nodes_to_explore = [action[1] for action in actions_to_explore if action[0] == 'village']
        actions_to_explore = self.generate_possible_actions_for_building_village(
                                                                                node=node,
                                                                                set_up=set_up
                                                                            )
        nodes_to_explore = [action[1] for action in actions_to_explore if action[0] == 'village']
        results = []
        for node in nodes_to_explore:
            temp_player = self.copy()
            temp_player.hand -= np.array(self.structure.real_estate_cost[1])
            temp_player.villages[node] = 1
            results.append((temp_player.calculate_value(), node))
            del temp_player
        results.sort(key=lambda a: a[0],reverse=True)
        return results
    
    def explore_building_town(self, node:int = None):
        """
        Explore building options for towns.
        Returns a list of tuples with the value and the node to build on.
        The list is ordered with the most optimal town first.

        Args:
            node (int, optional): Defaults to None.

        Returns:
            list[tuple]: A list of tuples with the value and the node to build on.
        """
        warnings.warn("This function is deprecated.", UserWarning)
        # if not node:
        #     nodes_to_explore = np.nonzero(self.villages)[0]
        # else:
        #     nodes_to_explore = [node]
        print('*')
        actions_to_explore = self.generate_possible_actions_for_building_town(node=node)
        print('*', len(actions_to_explore))
        nodes_to_explore = [action[1] for action in actions_to_explore if action[0] == 'town']
        results = []
        for node in nodes_to_explore:
            print('+')
            temp_player = self.copy()
            temp_player.hand -= np.array(self.structure.real_estate_cost[2])
            temp_player.villages[node] = 0
            temp_player.towns[node] = 1
            print('++')
            results.append((temp_player.calculate_value(), node))
            del temp_player
        results.sort(key=lambda a: a[0],reverse=True)
        return results
    
   
    def calculate_value_hand(self, hand_for_calculation: np.ndarray = None):
        """ Legacy, this function now refers to Py_Catan.ValueFunction.value_from_players_hand()"""
        warnings.warn("This function is deprecated.", UserWarning)
        if hand_for_calculation is None:
            hand_for_calculation = self.hand
        return self.value_function.value_from_players_hand(hand_for_calculation)
    
    def calculate_value_real_estate(self, 
                                    streets_for_calculation,
                                    villages_for_calculation,
                                    towns_for_calculation
                                    ):
        """ Legacy, this function now refers to Py_Catan.ValueFunction.value_from_players_real_estate()"""
        warnings.warn("This function is deprecated.", UserWarning)
        return self.value_function.value_from_players_real_estate(
            streets_for_calculation=streets_for_calculation, 
            villages_for_calculation=villages_for_calculation, 
            towns_for_calculation=towns_for_calculation
        )

    

    
    def calc_earning_power_for_player(self):
        ''' Legacy, this function now refers to Py_Catan.ValueFunction.calc_earning_power_for_player()'''
        warnings.warn("This function is deprecated.", UserWarning)
        self.earning_power = self.value_function.calc_earning_power_for_player(self) # potentially obsolete parameter!
        return self.earning_power
        # self.earning_power = np.sum(self.structure.node_earning_power[self.villages == 1],axis=0) + 2*np.sum(self.structure.node_earning_power[self.towns == 1],axis=0)
        # return self.earning_power
        
    def calc_earning_power_for_additional_village(self,extra_villages):
        ''' Legacy, this function now refers to Py_Catan.ValueFunction.calc_earning_power_for_additional_village()'''
        warnings.warn("This function is deprecated.", UserWarning)
        return self.value_function.calc_earning_power_for_additional_village(extra_villages)
    

        # This function calculates the earning power for an additional village
    #     return np.sum(self.structure.node_earning_power[extra_villages == 1],axis=0)

    # def generate_values_for_possible_actions(self, 
    #                       possible_actions: list,
    #                      ) -> list:
    #     '''
    #      ==== generate all options with values, then sort on order of possible actions and return a list with values ====
    #      ==== Not efficient but same interface as model based player ===
    #     Generate values for all possible actions.
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
    #     warnings.warn("This function is deprecated. Use player.generate_values_for_options() instead.", UserWarning)
    #     #
    #     #initial_value_hand = self.calculate_value_hand()
    #     #initial_value = self.calculate_value()
    #     #initial_value_real_estate = initial_value - initial_value_hand
    #     values = []
    #     for action in possible_actions:
    #         if action[0] == 'street':
    #             options = self.explore_building_street(edge=action[1])
    #             best_value = options[0][0]
    #             values.append(best_value)
    #         elif action[0] == 'village':
    #             options = self.explore_building_village(node=action[1])
    #             best_value = options[0][0]
    #             values.append(best_value)
    #         elif action[0] == 'town':
    #             options = self.explore_building_town(node=action[1])
    #             best_value = options[0][0]
    #             values.append(best_value)
    #         elif action[0] == 'trade_player':
    #             options = self.explore_trading_with_other_player(card_out_in=action[1])
    #             best_value = options[0][0] #+ initial_value_real_estate
    #             values.append(best_value)   
    #         elif action[0] == 'trade_specific_player':
    #             options = self.explore_trading_with_specific_player(card_out_in_partner=action[1][0])
    #             best_value = options[0][0] #+ initial_value_real_estate
    #             values.append(best_value)
    #         elif action[0] == 'trade_bank':
    #             options = self.explore_trading_with_bank(action[1])
    #             best_value = options[0][0] #+ initial_value_real_estate
    #             values.append(best_value)
    #         elif action[0] is None:
    #             # No action, just return the current value
    #             values.append(self.calculate_value())
    #         else:
    #             raise Exception(f'Unknown action type: {action[0]}')    
    #     return values
    
    # =============================================================================================

    # def find_best_action_old_method(self, 
    #                      rejected_trades_for_this_round: set = set([]),
    #                      rejected_trades_for_this_round_for_specific_player: set = set([])
    #                      ) -> tuple:
    #     """
    #     Method based on explore functions. This is deprecated and will be removed in the future.
        
    #     Return a single action as tuple e.g. ('street',0) or ('trade_player',(1,0))
    #     For this player the best action is decided based on the 'preferences'.

    #     The action types are:
    #     - 'street': build a street on the given edge 
    #     - 'village': build a village on the given node
    #     - 'town': build a town on the given node
    #     - 'trade_player': trade with another player
    #     - 'trade_specific_player': trade with a specific player
    #     - 'trade_bank': trade with the bank
    #     -   None: do not take any action, pass to next player

    #     Args:
    #         rejected_trades_for_this_round (set, optional). Defaults to set([]).
    #         rejected_trades_for_this_round_for_specific_player (set, optional). Defaults to set([]).

    #     Returns:
    #         tuple: action. 

    #     """
    #     #initial_value_hand = self.calculate_value_hand()
    #     initial_value = self.calculate_value()
    #     #initial_value_real_estate = initial_value - initial_value_hand
    #     best_value = initial_value
    #     best_action = (None,None)
    #     if self.can_build_street():
    #         options = self.explore_building_street()
    #         if options and options[0][0]  > best_value:
    #             best_value = options[0][0]
    #             best_action = ('street', options[0][1])
    #     if self.can_build_village():
    #         options = self.explore_building_village()
    #         if options and options[0][0] > best_value:
    #             best_value = options[0][0]
    #             best_action = ('village', options[0][1])
    #     if self.can_build_town():
    #         options = self.explore_building_town()
    #         if options and options[0][0] > best_value:
    #             best_value = options[0][0]
    #             best_action = ('town', options[0][1])
    #     if self.can_trade_with_player():
    #         options = self.explore_trading_with_other_player(rejected_trades=rejected_trades_for_this_round)
    #         if options and options[0][0]  > best_value:
    #             best_value = options[0][0]
    #             best_action = ('trade_player', options[0][1])
    #     if self.can_trade_with_specific_player():
    #         options = self.explore_trading_with_specific_player(
    #             rejected_trades_for_specific_player=rejected_trades_for_this_round,
    #             rejected_trades=rejected_trades_for_this_round_for_specific_player
    #             )
    #         if options and options[0][0]  > best_value:
    #             best_value = options[0][0]
    #             # second element of best_action is a tuple ( (card_out, card_in),trading partner )
    #             best_action = ('trade_specific_player', options[0][1])
    #     if self.can_trade_with_bank():
    #         options = self.explore_trading_with_bank()
    #         if options and options[0][0]  > best_value:
    #             best_value = options[0][0]
    #             best_action = ('trade_bank', options[0][1]) 
    #     return best_action
 
    
    
  
 