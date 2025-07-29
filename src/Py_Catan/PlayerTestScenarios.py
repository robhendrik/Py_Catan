from Py_Catan.Preferences import PlayerPreferences
from Py_Catan.BoardStructure import BoardStructure
from Py_Catan.Player import Player

import numpy as np
import random

class TestPlayer(Player):
    '''
    A player only used for tests. Trades are always rejected.
    If this is the first player in the round he will build a town,
    if it is the second player he will build a village,
    if it is the third player he will build a street.
    If it is the fourth player he will not build anything.
    In the setup phase this player will not build anything.
    '''
    def __init__(self,
                 name: str = 'A', 
                 structure: BoardStructure = BoardStructure(),
                 preference: any = PlayerPreferences(), 
                 status: dict = dict([])):
        # Call to Player's constructor
        super().__init__(
                name = name, 
                structure = structure,
                preference = preference, 
                status = status
                )

    def copy(self) -> 'TestPlayer':
        '''
        Create a copy of the player with the same attributes.
        The name is set to 'New' to avoid conflicts with the original player.
        '''
        new_player = TestPlayer(name ='New')
        np = vars(new_player)
        for k,v in vars(self).items():
            if k != 'name':
                try:
                    np[k] = v.copy()
                except:
                    np[k] = v
        return new_player
    
    def respond_positive_to_other_players_trading_request(self,card_out_in) -> bool:
        """
        This player always reject trades.
        """
        return False
    
    def find_best_action(self,
                         rejected_trades_for_this_round: set = set([]),
                         rejected_trades_for_this_round_for_specific_player: set = set([])
                         ) -> tuple:
        """
        Decide action for this player.
        """
        # the position indicates whether you are first, second etc player
        position = self._players_in_this_game.index(self) 
        
        # find free nodes and edges to build on
        # if none this must be the first turn
        # can also be that board is full, this scenario will lead to error!
        nodes_to_explore = np.nonzero(self.build_options['village_options'])[0]
        if len(nodes_to_explore) == 0:
            nodes_to_explore = np.nonzero(self.free_nodes_on_board)[0]
        edges_to_explore = np.nonzero(self.build_options['street_options'])[0]
        if len(edges_to_explore) == 0:
            edges_to_explore = np.nonzero(self.free_edges_on_board)[0]
        # take a single action based on the position of the player
        if position == 0 and self._actions_in_round == 0:
            self.hand =np.array(self.structure.real_estate_cost[1])
            action =  ('village', nodes_to_explore[0])
        elif position == 0 and self._actions_in_round == 1:
            self.hand =np.array(self.structure.real_estate_cost[2])
            action =  ('town', np.nonzero(self.villages)[0])
        elif position == 1 and self._actions_in_round == 0:
            self.hand =np.array(self.structure.real_estate_cost[1])
            action = ('village', nodes_to_explore[0])
        elif position == 2 and self._actions_in_round == 0:
            self.hand =np.array(self.structure.real_estate_cost[0])
            action =  ('street', edges_to_explore[0])
        else:
            action = (None, None)
        return action
    
    def player_setup(self,brd) -> None:
        """
        Setup the player with initial buildings.
        This player builds nothing.
        """
        return [(None, None), (None, None)]
        

   

    
   
    
    
    

    
    
  
 