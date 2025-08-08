from Py_Catan.Preferences import PlayerPreferences
from Py_Catan.BoardStructure import BoardStructure
from Py_Catan.PlayerModelBased import Player_Model_Based
import numpy as np
import random
from functools import cache
from Py_Catan.Player import Player

class Player_Model_Based_With_Randomness(Player_Model_Based):
    def __init__(self,
                 name: str = 'A', 
                 structure: BoardStructure = BoardStructure(),
                 fraction_of_randomness: float = 0.0,
                 status: dict = dict([]) ):
        # Call to Player's constructor
        super().__init__(
                name = name, 
                structure = structure,
                status = status
                )
        self.fraction_of_randomness = fraction_of_randomness

        
    def copy(self) -> 'Player_Model_Based_With_Randomness':
        '''
        Create a copy of the player with the same attributes.
        The name is set to 'New' to avoid conflicts with the original player.
        '''
        #new_player = Player_Model_Based(name ='New',structure=self.structure, preference=self.preference)
        new_player = Player_Model_Based_With_Randomness(name ='New')
        np = vars(new_player)
        for k,v in vars(self).items():
            if k != 'name':
                try:
                    np[k] = v.copy()
                except:
                    np[k] = v
        return new_player
    
    def find_best_action(self, 
                         rejected_trades_for_this_round: set = set([]),
                         rejected_trades_for_this_round_for_specific_player: set = set([])
                         ) -> tuple:
        '''
        Find the best action for the player based on the current state of the board.
        Based on player.fraction_of_randomness, the player will either use the original method or select 
        a random action from the possible actions.
        '''
        random_value = np.random.random(1)[0]
        if self.fraction_of_randomness == 0.0 or random_value > self.fraction_of_randomness:
            # If no randomness is applied, use the original method
            return super().find_best_action(rejected_trades_for_this_round, rejected_trades_for_this_round_for_specific_player)
        else:
            # If randomness is applied, select a random action from the possible actions
            possible_actions = self.generate_list_of_possible_actions(rejected_trades=rejected_trades_for_this_round)
            if not possible_actions:
                return (None, None)
            # Select a random action from the possible actions
            best_index = np.random.randint(0, len(possible_actions))
            best_action = possible_actions[best_index]
            return best_action
        
    def player_setup(self,brd) -> None:
        '''
        Setup the player with initial buildings. Which streets and villages are built is based on the preferences.

        Function returns a list of actions to be executed by the board.
        The actions are tuples of the form ('street', edge) or ('village', node)
        '''
        random_value = np.random.random(1)[0]
        if self.fraction_of_randomness == 0.0 or random_value > self.fraction_of_randomness:
            # If no randomness is applied, use the original method
            return super().player_setup(brd)
        else:
            self.update_build_options()
            self.hand = np.array(brd.structure.real_estate_cost[1])
            options = self.explore_building_village(set_up=True)
            if len(options) == 0:
                raise Exception("No options available for building a village")
            random_index = np.random.randint(0, len(options))
            random_node = options[random_index][1]
            self.update_build_options()
            self.hand += np.array(brd.structure.real_estate_cost[0])
            options = self.explore_building_street(set_up=True, set_up_node=random_node)
            if len(options) == 0:
                raise Exception("No options available for building a street")
            random_index = np.random.randint(0, len(options))
            
            random_edge = options[random_index][1]
            return [('village', random_node),('street',random_edge)]

       
    

    
    
    

    
 