from Py_Catan.Preferences import PlayerPreferences
from Py_Catan.BoardStructure import BoardStructure
from Py_Catan.Player import Player

import numpy as np
import random

class Player_Passive(Player):
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

    def copy(self) -> 'Player_Passive':
        '''
        Create a copy of the player with the same attributes.
        The name is set to 'New' to avoid conflicts with the original player.
        '''
        new_player = Player_Passive(name ='New')
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
                         ) -> tuple    :
        return (None,None)
    
    def player_setup(self,brd) -> None:
        '''
        Setup the player with initial buildings.
        '''
        return [(None,None),(None,None)]
        

   

    
   
    
    
    

    
    
  
 