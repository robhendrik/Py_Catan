from Py_Catan.Preferences import PlayerPreferences
from Py_Catan.BoardStructure import BoardStructure
from Py_Catan.Player import Player
import numpy as np
import random

class Player_Random(Player):
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
        self.threshold_for_accepting_trade = 0.5
        self.max_actions_in_round = 5

    def copy(self) -> 'Player_Random':
        '''
        Create a copy of the player with the same attributes.
        The name is set to 'New' to avoid conflicts with the original player.
        '''
        new_player = Player_Random(name ='New')
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

        return np.random.random() < self.threshold_for_accepting_trade
    
    def find_best_action(self,
                         rejected_trades_for_this_round: set = set([]),
                         rejected_trades_for_this_round_for_specific_player: set = set([])
                         ) -> tuple:
        '''
        If can build, build something random.
        Otherwise try to trade for stronger hand.
        After attemps retrun (None, None).
        if np.random.random() < self.threshold_for_skipping_turn:
            return (None,None)
        '''
        if self._actions_in_round >= self.max_actions_in_round:
            return (None,None)
        # create build options if possible
        build_options = []
        if self.can_build_street():
            build_options += self.explore_building_street()
        if self.can_build_village():
            build_options += self.explore_building_village()
        if self.can_build_town():
            build_options += self.explore_building_town()
        # create trade options if possible
        if self.can_trade_with_player():
            trade_options = self.explore_trading_with_other_player(rejected_trades=rejected_trades_for_this_round)
        else:   
            trade_options = [(None,None)]
        if self.can_trade_with_bank:
            bank_options = self.explore_trading_with_bank()
        else:
            bank_options = [(None,None)]
        # execute options
        if build_options:
            return random.choice(build_options)
        elif trade_options[0][0] == 'trade_player':
            return trade_options[0]
        elif bank_options[0][0] == 'trade_bank':
            return bank_options[0]
        else:
            return (None,None)

   
    def explore_trading_with_other_player(self,rejected_trades: set = None,card_out_in: tuple = None):
        if not card_out_in:
            swaps_to_explore =   [
                (card_out,card_in) for card_out in range(len(self.hand))
                    for card_in in range(len(self.hand)) if 
                    (card_out != card_in and self.hand[card_out] > 0 and (card_out,card_in) not in rejected_trades)
                    ]
        else:
            swaps_to_explore = [(card_out_in[0],card_out_in[1])]
        best_value = 0
        best_swap = None
        for card_out,card_in in swaps_to_explore:
            temp_hand = self.hand.copy()
            temp_hand[card_out] -= 1
            temp_hand[card_in] += 1
            if self.calculate_value_hand(temp_hand) > best_value:
                best_value = self.calculate_value_hand(temp_hand)
                best_swap = (card_out,card_in)
        if best_value > 0:
            return [('trade_player',best_swap)] 
        else:
            return [(None,None)]
        
    def explore_trading_with_bank(self,card_out_in: tuple = None):
        if not card_out_in:
            swaps_to_explore =   [(card_out,card_in) for card_out in range(len(self.hand))
                    for card_in in range(len(self.hand)) if (card_out != card_in and self.hand[card_out] >= 4)]
        else:
            swaps_to_explore = [(card_out_in[0],card_out_in[1])]
        best_value = 0
        best_swap = None
        for card_out,card_in in swaps_to_explore:
            temp_hand = self.hand.copy()
            temp_hand[card_out] -= 4
            temp_hand[card_in] += 1
            if self.calculate_value_hand(temp_hand) > best_value:
                best_value = self.calculate_value_hand(temp_hand)
                best_swap = (card_out,card_in)
        if best_value > 0:
            return [('trade_bank',best_swap)] 
        else:
            return [(None,None)]


    def explore_building_street(self,edge: tuple = None, set_up: bool = False, set_up_node: int = None):
        if not set_up:
            if not edge:
                edges_to_explore = np.nonzero(self.build_options['street_options'])[0]
            else:
                edges_to_explore = [edge]
        else:
            edges_to_explore = [edge for edge,connecting_nodes in enumerate(self.structure.nodes_connected_by_edge) if set_up_node in connecting_nodes]
        return [('street',edge) for edge in edges_to_explore]
    
    def explore_building_village(self, node: int = None, set_up: bool = False):
        if not set_up:
            if not node:
                nodes_to_explore = np.nonzero(self.build_options['village_options'])[0]
            else:
                nodes_to_explore = [node]
        else:
            nodes_to_explore = np.nonzero(self.free_nodes_on_board)[0]
        return [('village',node) for node in nodes_to_explore]
    
    def explore_building_town(self, node:int = None):
        if not node:
            nodes_to_explore = np.nonzero(self.villages)[0]
        else:
            nodes_to_explore = [node]

        return [('town',node) for node in nodes_to_explore]
    
    def player_setup(self,brd) -> None:
        '''
        Setup the player with initial buildings.
        '''
        self.update_build_options()
        self.hand = np.array(self.structure.real_estate_cost[1])
        options = self.explore_building_village(set_up=True)
        if options:
            best_node = ('village',random.choice(options)[1])
        else:
            best_node = (None,None)
        self.update_build_options()
        self.hand += np.array(self.structure.real_estate_cost[0])
        options = self.explore_building_street(set_up=True, set_up_node=best_node[1])
        if options:
            best_edge = ('street',random.choice(options)[1])
        else:
            best_edge = (None,None)
        return [best_node,best_edge]
    
    def calculate_value_hand(self, hand_for_calculation: np.ndarray = None):
        '''
        Calculate the value of the player's hand
        '''
        if hand_for_calculation is None:
            hand_for_calculation = self.hand
   
        value = np.all( hand_for_calculation >= self.structure.real_estate_cost[0]) * 10
        value += np.all( hand_for_calculation >= self.structure.real_estate_cost[1]) * 20
        value += np.all( hand_for_calculation >= self.structure.real_estate_cost[2]) * 30
        # value of secondary options
        helper = ( hand_for_calculation- np.array(self.structure.real_estate_cost[0]))
        value += (1 if -2 == np.sum(helper[helper < 0]) else 0) * 5
        helper = ( hand_for_calculation - np.array(self.structure.real_estate_cost[1]))
        value += (1 if -2 == np.sum(helper[helper < 0]) else 0)  * 11
        helper = ( hand_for_calculation - np.array(self.structure.real_estate_cost[2]))
        value += (1 if -2 == np.sum(helper[helper < 0]) else 0)  * 15
        # value of tertiary options
        helper = ( hand_for_calculation - np.array(self.structure.real_estate_cost[1]))
        value += (1 if -2 == np.sum(helper[helper < 0]) else 0) * 2
        helper = ( hand_for_calculation - np.array(self.structure.real_estate_cost[2]))
        value += (1 if -2 == np.sum(helper[helper < 0]) else 0)  * 3

        return value
    
    def calculate_value(self, updated_version: bool = False):
        if not hasattr(self, 'preference'):
            value = 0
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
    
    
    

    
    
  
 