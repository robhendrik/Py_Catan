from Py_Catan.Preferences import PlayerPreferences
from Py_Catan.BoardStructure import BoardStructure
from Py_Catan.Player import Player
from Py_Catan.Board import Board
from Py_Catan.BoardVector import BoardVector
import numpy as np
import warnings

class ValueFunction:
    def __init__(self, preference: PlayerPreferences, structure: BoardStructure, player: Player = None):
        self.preference = preference
        self.structure = structure
        self.player = player
        return
    
    def value_for_board(self, board = Board) -> np.ndarray:
        board._update_board_for_players()
        values = []
        for player in board.players:
            player.update_build_options()
            values.append(self.value_for_player(player))
        return np.array(values,np.float32)
    
    def value_for_board_vector(self, board_vector: BoardVector) -> np.ndarray:
        board = board_vector.create_board_from_vector(list_of_players=board_vector.players)
        return self.value_for_board(board)
    
    def value_from_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        Calculate the value of a board from a vector representation.
        This method creates a new Board instance from the vector and calculates the value for each player.
        
        The players are created based on the type of the player stored in self.player, or a default Player if self.player is None.

        Args:
            vector (np.ndarray): The vector representation of the board.
        
        Returns:
            np.ndarray: An array of values for each player based on the vector.
        """

        # Create new instances of the same type as self.player (if available), otherwise use default Player
        player_type = type(self.player) if self.player is not None else Player
        # value based player is not yet defined here, so we have to do an ugly trick to avoid circular import
        try:
            players = [player_type(name=f'Player_{i+1}', structure=self.structure, preference=self.preference) for i in range(4)]
        except:
            players = [player_type(name=f'Player_{i+1}', structure=self.structure) for i in range(4)]   
        board = Board(structure=self.structure)
        board.players = players
        board_vector = BoardVector(board=board) #board_vector.players should now be equal to players
        board_vector.vector = vector
        return self.value_for_board_vector(board_vector)
    
    def value_for_player(self, player: Player) -> float:
        """
        Calculate the value for a specific player based on their current state and preferences.

        Args:
            player (Player): The player for whom the value is calculated.

        Returns:
            float: The calculated value for the player.
        """
        if player._players_in_this_game == [] or player._board is None or player._player_position is None:
            warnings.warn("We are using ValueFunction.value_for_player() on a player that is not properly assigned to a board.", UserWarning)
        value = 0.0
        # calculate score
        score = 0
        score += sum(player.towns) * 2
        score += sum(player.villages) * 1
        score += 2 if  player.owns_longest_street else 0
        value += self.preference.full_score if score==self.structure.winning_score else 0
        
        # value of direct posessions
        value += np.sum(player.streets) * self.preference.streets
        value += np.sum(player.villages) * self.preference.villages
        value += np.sum(player.towns) * self.preference.towns
        
        # value of cards in hand and penalty for too many cards
        penalty_factor = (sum(player.hand)/(sum(player.hand)+self.preference.penalty_reference_for_too_many_cards) )
        value += penalty_factor * np.inner(player.hand,self.preference.resource_type_weight) * self.preference.cards_in_hand

        # value of current earning power
        earning_power = np.sum(self.structure.node_earning_power[player.villages == 1],axis=0) + 2*np.sum(self.structure.node_earning_power[player.towns == 1],axis=0)
        value += np.dot(earning_power ,self.preference.resource_type_weight) * self.preference.cards_earning_power

        # value of direct options
        value += np.all(player.hand >= self.structure.real_estate_cost[0]) * self.preference.hand_for_street
        value += np.all(player.hand >= self.structure.real_estate_cost[1]) * self.preference.hand_for_village
        value += np.all(player.hand >= self.structure.real_estate_cost[2]) * self.preference.hand_for_town
        value += np.sum(player.build_options['street_options']) * self.preference.street_build_options
        value += np.sum(player.build_options['village_options']) * self.preference.village_build_options

        # value of earning power for direct options
        extra_villages=player.build_options['village_options']
        secondary_earning_power =  np.sum(self.structure.node_earning_power[extra_villages == 1],axis=0)
        value += np.dot(secondary_earning_power ,self.preference.resource_type_weight) * self.preference.direct_options_earning_power
        
        # value of secondary options
        helper = (player.hand - np.array(self.structure.real_estate_cost[0]))
        value += (1 if -2 == np.sum(helper[helper < 0]) else 0) * self.preference.hand_for_street_missing_one
        helper = (player.hand - np.array(self.structure.real_estate_cost[1]))
        value += (1 if -2 == np.sum(helper[helper < 0]) else 0)  * self.preference.hand_for_village_missing_one
        helper = (player.hand - np.array(self.structure.real_estate_cost[2]))
        value += (1 if -2 == np.sum(helper[helper < 0]) else 0)  * self.preference.hand_for_town_missing_one
        value += np.sum(player.build_options['secondary_village_options']) * self.preference.secondary_village_build_options
    
        # value of secondary options earning power
        extra_villages=player.build_options['secondary_village_options']
        secondary_earning_power =  np.sum(self.structure.node_earning_power[extra_villages == 1],axis=0)
        value += np.dot(secondary_earning_power ,self.preference.resource_type_weight) * self.preference.secondary_options_earning_power

        # value of tertiary options
        helper = (player.hand - np.array(self.structure.real_estate_cost[1]))
        value += (1 if -2 == np.sum(helper[helper < 0]) else 0) * self.preference.hand_for_village_missing_two
        helper = (player.hand - np.array(self.structure.real_estate_cost[2]))
        value += (1 if -2 == np.sum(helper[helper < 0]) else 0)  * self.preference.hand_for_town_missing_two

        return value
    
    # ====================================================================================
    #
    # Deprecated methods, kept for backward compatibility
    #
    # ====================================================================================
    def value_from_players_hand(self, hand_for_calculation: np.ndarray = None):
        '''
        Calculate the value of the player's hand based on the preferences.
        If hand_for_calculation is provided, it will be used instead of the player's hand.
        '''
        warnings.warn("This function is deprecated.", UserWarning)
        if hand_for_calculation is None:
            hand_for_calculation = self.player.hand
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
    
    def value_from_players_real_estate(self, 
                                        streets_for_calculation,
                                        villages_for_calculation,
                                        towns_for_calculation
                                        ):
        '''
        Calculate the value of the player's real estate based on the preferences.
        '''
        warnings.warn("This function is deprecated.", UserWarning)
        if streets_for_calculation is None:
            streets = self.player.streets
        if villages_for_calculation is None:
            villages = self.player.villages
        if towns_for_calculation is None:
            towns = self.player.towns
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
    
    def calc_earning_power_for_player(self,player: Player):
        warnings.warn("This function is deprecated.", UserWarning)
        return np.sum(self.structure.node_earning_power[player.villages == 1],axis=0) + 2*np.sum(self.structure.node_earning_power[player.towns == 1],axis=0)
    
    def calc_earning_power_for_additional_village(self,extra_villages):
        warnings.warn("This function is deprecated.", UserWarning)
        return np.sum(self.structure.node_earning_power[extra_villages == 1],axis=0)
