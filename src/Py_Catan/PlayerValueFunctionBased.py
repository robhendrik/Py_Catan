from Py_Catan.Preferences import PlayerPreferences
from Py_Catan.BoardStructure import BoardStructure
from Py_Catan.Player import Player
import numpy as np

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
                preference = preference, 
                status = status
                )
    def copy(self) -> 'Player_Value_Function_Based':
        '''
        Create a copy of the player with the same attributes.
        The name is set to 'New' to avoid conflicts with the original player.
        '''
        #new_player = Player_Value_Function_Based(name ='New',structure=self.structure, preference=self.preference)
        new_player = Player_Value_Function_Based(name ='New')
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
        current_value = self.calculate_value_hand(self.hand)
        temp_hand = self.hand.copy()
        temp_hand[card_out_in[0]] -= 1
        temp_hand[card_out_in[1]] += 1
        new_value = self.calculate_value_hand(temp_hand)
        return (new_value > current_value) and not np.isclose(new_value,current_value, atol=self.atol)
    
    def explore_trading_with_specific_player(self,
                                              rejected_trades: set = None,
                                              rejected_trades_for_specific_player: set = None,
                                              card_out_in: tuple = None):
        results = []
        # results will be a list of tuples [( value, ((card_out, card_in),trading_partner)) ), ...]
        return results
        
    def explore_trading_with_other_player(self,rejected_trades: set = None,card_out_in: tuple = None):
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
            temp_hand = self.hand.copy()
            temp_hand[card_out] -= 1
            temp_hand[card_in] += 1
            results.append((self.calculate_value_hand(temp_hand), (card_out,card_in)))
        results.sort(key=lambda a: a[0],reverse=True)
        return results
        
    def explore_trading_with_bank(self,card_out_in: tuple = None):
        if not card_out_in:
            swaps_to_explore =   [(card_out,card_in) for card_out in range(len(self.hand))
                    for card_in in range(len(self.hand)) if (card_out != card_in and self.hand[card_out] >= 4)]
        else:
            swaps_to_explore = [(card_out_in[0],card_out_in[1])]
        results = []
        for card_out,card_in in swaps_to_explore:
            temp_hand = self.hand.copy()
            temp_hand[card_out] -= 4
            temp_hand[card_in] += 1
            results.append((self.calculate_value_hand(temp_hand), (card_out,card_in)))
        results.sort(key=lambda a: a[0],reverse=True)
        return results

    def explore_building_street(self,edge: tuple = None, set_up: bool = False, set_up_node: int = None):
        if not set_up:
            if not edge:
                edges_to_explore = np.nonzero(self.build_options['street_options'])[0]
            else:
                edges_to_explore = [edge]
        else:
            edges_to_explore = [edge for edge,connecting_nodes in enumerate(self.structure.nodes_connected_by_edge) if set_up_node in connecting_nodes]
        results = []
        for edge in edges_to_explore:
            temp_player = self.copy()
            #temp_player.build_street(edge)
            temp_player.hand -= np.array(temp_player.structure.real_estate_cost[0])
            temp_player.streets[edge] = 1
            temp_player.calculate_longest_street()
            temp_player.update_build_options()
            results.append((temp_player.calculate_value(), edge))
            del temp_player
        results.sort(key=lambda a: a[0],reverse=True)
        return results
    
    def explore_building_village(self, node: int = None, set_up: bool = False):
        if not set_up:
            if not node:
                nodes_to_explore = np.nonzero(self.build_options['village_options'])[0]
            else:
                nodes_to_explore = [node]
        else:
            nodes_to_explore = np.nonzero(self.free_nodes_on_board)[0]
       
        results = []
        for node in nodes_to_explore:
            temp_player = self.copy()
            temp_player.hand -= np.array(self.structure.real_estate_cost[1])
            temp_player.villages[node] = 1
            temp_player.earning_power = temp_player.calc_earning_power_for_player()
            results.append((temp_player.calculate_value(), node))
            del temp_player
        results.sort(key=lambda a: a[0],reverse=True)
        return results
    
    def explore_building_town(self, node:int = None):
        if not node:
            nodes_to_explore = np.nonzero(self.villages)[0]
        else:
            nodes_to_explore = [node]
        results = []
        for node in nodes_to_explore:
            temp_player = self.copy()
            temp_player.hand -= np.array(self.structure.real_estate_cost[2])
            temp_player.villages[node] = 0
            temp_player.towns[node] = 1
            temp_player.earning_power = temp_player.calc_earning_power_for_player()
            results.append((temp_player.calculate_value(), node))
            del temp_player
        results.sort(key=lambda a: a[0],reverse=True)
        return results
    
   
    def calculate_value_hand(self, hand_for_calculation: np.ndarray = None):
        '''
        Calculate the value of the player's hand based on the preferences.
        If hand_for_calculation is provided, it will be used instead of the player's hand.
        '''
        if hand_for_calculation is None:
            hand_for_calculation = self.hand
        penalty_factor = (sum(hand_for_calculation)/(sum(hand_for_calculation)+self.preference.penalty_reference_for_too_many_cards) )
        value = penalty_factor * np.inner(hand_for_calculation,self.preference.resource_type_weight) * self.preference.cards_in_hand
        value += np.all( hand_for_calculation >= self.structure.real_estate_cost[0]) * self.preference.hand_for_street
        value += np.all( hand_for_calculation >= self.structure.real_estate_cost[1]) * self.preference.hand_for_village
        value += np.all( hand_for_calculation >= self.structure.real_estate_cost[2]) * self.preference.hand_for_town
        # value of secondary options
        helper = ( hand_for_calculation- np.array(self.structure.real_estate_cost[0]))
        value += (1 if -2 == np.sum(helper[helper < 0]) else 0) * self.preference.hand_for_street_missing_one
        helper = ( hand_for_calculation - np.array(self.structure.real_estate_cost[1]))
        value += (1 if -2 == np.sum(helper[helper < 0]) else 0)  * self.preference.hand_for_village_missing_one
        helper = ( hand_for_calculation - np.array(self.structure.real_estate_cost[2]))
        value += (1 if -2 == np.sum(helper[helper < 0]) else 0)  * self.preference.hand_for_town_missing_one
        # value of tertiary options
        helper = ( hand_for_calculation - np.array(self.structure.real_estate_cost[1]))
        value += (1 if -2 == np.sum(helper[helper < 0]) else 0) * self.preference.hand_for_village_missing_two
        helper = ( hand_for_calculation - np.array(self.structure.real_estate_cost[2]))
        value += (1 if -2 == np.sum(helper[helper < 0]) else 0)  * self.preference.hand_for_town_missing_two

        return value
    
    def calculate_value_real_estate(self, 
                                    streets_for_calculation,
                                    villages_for_calculation,
                                    towns_for_calculation
                                    ):
        '''
        Calculate the value of the player's real estate based on the preferences.
        '''
        if streets_for_calculation is None:
            streets = self.streets
        if villages_for_calculation is None:
            villages = self.villages
        if towns_for_calculation is None:
            towns = self.towns
        value = 0
        # value of direct posessions
        value += np.sum(streets_for_calculation) * self.preference.streets
        value += np.sum(villages_for_calculation) * self.preference.villages
        value += np.sum(towns_for_calculation) * self.preference.towns
        
       # value of current earning power
        value += np.dot(self.earning_power ,self.preference.resource_type_weight) * self.preference.cards_earning_power

        # value of direct options
        value += np.sum(self.build_options['street_options']) * self.preference.street_build_options
        value += np.sum(self.build_options['village_options']) * self.preference.village_build_options       
        
        # value of earning power for direct options
        secondary_earning_power = self.calc_earning_power_for_additional_village(
            extra_villages=self.build_options['village_options'])
        value += np.dot(secondary_earning_power ,self.preference.resource_type_weight) * self.preference.direct_options_earning_power
        
        # value of secondary options
        value += np.sum(self.build_options['secondary_village_options']) * self.preference.secondary_village_build_options
    
        # value of secondary options earning power
        secondary_earning_power = self.calc_earning_power_for_additional_village(
            extra_villages=self.build_options['secondary_village_options'])
        value += np.dot(secondary_earning_power ,self.preference.resource_type_weight) * self.preference.secondary_options_earning_power

        
        return value
    
    def calculate_value(self, updated_version: bool = False):
        if updated_version:
            # This is the updated version of the value calculation
            # It uses the current state of the player and the board
            value = 0
            # value of winning
            value += self.preference.full_score if self.calculate_score()==self.structure.winning_score else 0
            # value of hand
            value += self.calculate_value_hand() 
            # value of real estate
            value += self.calculate_value_real_estate(self.streets, self.villages, self.towns)
            return value
        else:
            # first run board.update_board_for_players and board.build_options(player)
            value = 0

            # value of winning
            value += self.preference.full_score if self.calculate_score()==self.structure.winning_score else 0
            
            # value of direct posessions
            value += np.sum(self.streets) * self.preference.streets
            value += np.sum(self.villages) * self.preference.villages
            value += np.sum(self.towns) * self.preference.towns
            
            # value of cards in hand and penalty for too many cards
            penalty_factor = (sum(self.hand)/(sum(self.hand)+self.preference.penalty_reference_for_too_many_cards) )
            value += penalty_factor * np.inner(self.hand,self.preference.resource_type_weight) * self.preference.cards_in_hand
            
            # value of current earning power
            value += np.dot(self.earning_power ,self.preference.resource_type_weight) * self.preference.cards_earning_power

            # value of direct options
            value += np.all(self.hand >= self.structure.real_estate_cost[0]) * self.preference.hand_for_street
            value += np.all(self.hand >= self.structure.real_estate_cost[1]) * self.preference.hand_for_village
            value += np.all(self.hand >= self.structure.real_estate_cost[2]) * self.preference.hand_for_town
            value += np.sum(self.build_options['street_options']) * self.preference.street_build_options
            value += np.sum(self.build_options['village_options']) * self.preference.village_build_options

            # value of earning power for direct options
            secondary_earning_power = self.calc_earning_power_for_additional_village(
                extra_villages=self.build_options['village_options'])
            value += np.dot(secondary_earning_power ,self.preference.resource_type_weight) * self.preference.direct_options_earning_power
            
            # value of secondary options
            helper = (self.hand - np.array(self.structure.real_estate_cost[0]))
            value += (1 if -2 == np.sum(helper[helper < 0]) else 0) * self.preference.hand_for_street_missing_one
            helper = (self.hand - np.array(self.structure.real_estate_cost[1]))
            value += (1 if -2 == np.sum(helper[helper < 0]) else 0)  * self.preference.hand_for_village_missing_one
            helper = (self.hand - np.array(self.structure.real_estate_cost[2]))
            value += (1 if -2 == np.sum(helper[helper < 0]) else 0)  * self.preference.hand_for_town_missing_one
            value += np.sum(self.build_options['secondary_village_options']) * self.preference.secondary_village_build_options
        
            # value of secondary options earning power
            secondary_earning_power = self.calc_earning_power_for_additional_village(
                extra_villages=self.build_options['secondary_village_options'])
            value += np.dot(secondary_earning_power ,self.preference.resource_type_weight) * self.preference.secondary_options_earning_power

            # value of tertiary options
            helper = (self.hand - np.array(self.structure.real_estate_cost[1]))
            value += (1 if -2 == np.sum(helper[helper < 0]) else 0) * self.preference.hand_for_village_missing_two
            helper = (self.hand - np.array(self.structure.real_estate_cost[2]))
            value += (1 if -2 == np.sum(helper[helper < 0]) else 0)  * self.preference.hand_for_town_missing_two
            
            return value
    
    def calc_earning_power_for_player(self):
        self.earning_power = np.sum(self.structure.node_earning_power[self.villages == 1],axis=0) + 2*np.sum(self.structure.node_earning_power[self.towns == 1],axis=0)
        return self.earning_power
        
    def calc_earning_power_for_additional_village(self,extra_villages):
        return np.sum(self.structure.node_earning_power[extra_villages == 1],axis=0)
    
    def find_best_action(self, 
                         rejected_trades_for_this_round: set = set([]),
                         rejected_trades_for_this_round_for_specific_player: set = set([])
                         ) -> tuple:
        initial_value_hand = self.calculate_value_hand()
        initial_value = self.calculate_value()
        initial_value_real_estate = initial_value - initial_value_hand
        best_value = initial_value
        best_action = (None,None)
        if self.can_build_street():
            options = self.explore_building_street()
            if options and options[0][0]  > best_value:
                best_value = options[0][0]
                best_action = ('street', options[0][1])
        if self.can_build_village():
            options = self.explore_building_village()
            if options and options[0][0] > best_value:
                best_value = options[0][0]
                best_action = ('village', options[0][1])
        if self.can_build_town():
            options = self.explore_building_town()
            if options and options[0][0] > best_value:
                best_value = options[0][0]
                best_action = ('town', options[0][1])
        if self.can_trade_with_player():
            options = self.explore_trading_with_other_player(rejected_trades=rejected_trades_for_this_round)
            if options and options[0][0] + initial_value_real_estate > best_value:
                best_value = options[0][0]
                best_action = ('trade_player', options[0][1])
        if self.can_trade_with_specific_player():
            options = self.explore_trading_with_specific_player(
                rejected_trades_for_specific_player=rejected_trades_for_this_round,
                rejected_trades=rejected_trades_for_this_round_for_specific_player
                )
            if options and options[0][0] + initial_value_real_estate > best_value:
                best_value = options[0][0]
                # second element of best_action is a tuple ( (card_out, card_in),trading partner )
                best_action = ('trade_specific_player', options[0][1])
        if self.can_trade_with_bank():
            options = self.explore_trading_with_bank()
            if options and options[0][0] + initial_value_real_estate > best_value:
                best_value = options[0][0]
                best_action = ('trade_bank', options[0][1]) 
        return best_action
    
    
    def player_setup(self,brd) -> None:
        '''
        Setup the player with initial buildings.
        '''
        self.update_build_options()
        self.hand = np.array(brd.structure.real_estate_cost[1])
        options = self.explore_building_village(set_up=True)
        best_node = options[0][1]
        self.update_build_options()
        self.hand += np.array(brd.structure.real_estate_cost[0])
        options = self.explore_building_street(set_up=True, set_up_node=best_node)
        best_edge = options[0][1]
        return [('village', best_node),('street',best_edge)]
    
    
  
 