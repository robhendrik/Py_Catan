import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass, field, asdict

@dataclass
class BoardLayout:
    """
    BoardLayout defines the configuration and rules for a game board.   
    This is a dataclass to define the layout of the board.
    It contains the tile layout, values for each tile, 
    scale, rings, costs for building streets, villages, towns and development cards,
    winning score, dice value to hand in cards, maximum available villages, 
    towns and streets,  and the minimum length for the longest street.  

    Attributes:
        tile_layout (str): String representing the arrangement of tiles on the board.
        values (list): List of integers representing the value assigned to each tile.
        scale (int): Scale factor for the board layout.
        rings (int): Number of concentric rings in the board.
        street_cost (str): Resource cost to build a street.
        village_cost (str): Resource cost to build a village.
        town_cost (str): Resource cost to build a town.
        development_card_cost (str): Resource cost to buy a development card.
        winning_score (str): Score required to win the game.
        dice_value_to_hand_in_cards (str): Dice value at which players must hand in cards.
        max_available_villages (str): Maximum number of villages a player can build.
        max_available_towns (str): Maximum number of towns a player can build.
        max_available_streets (str): Maximum number of streets a player can build.
        longest_street_minimum (int): Minimum length required for the longest street.
    
    Methods:
        asdict():
            Returns a dictionary representation of the BoardLayout instance.
        copy():
            Returns a deep copy of the BoardLayout instance.
    """
    tile_layout: str = 'DSWSWSWWGSOBGBGOBOG'
    values: list = field(default_factory=lambda: [0,11,3,6,5,4, 9,10,8,4,11,12,9,10,8,3,6,2,5])
    scale: int = 5
    rings: int = 3
    resource_types: str = 'BDGOSW'
    street_cost: str = 'BW'
    village_cost: str = 'BGSW'
    town_cost: str = 'GGOOO'
    development_card_cost: str  = 'GOS'
    winning_score: str = 10
    dice_value_to_hand_in_cards: str = 7
    max_available_villages: str = 5
    max_available_towns: str = 5
    max_available_streets: str = 12
    longest_street_minimum: int = 3
    plot_colors_players = ['blue','green','red','yellow','purple','pink','orange']
    plot_labels_for_resources = ['Brick', 'Desert','Grain', 'Ore', 'Sheep', 'Wood']
    plot_max_card_in_hand_per_type = 7

    def asdict(self):
        '''
        Returns a dictionary representation of the BoardLayout instance.
        This method is useful for serialization or when passing the instance data to other functions.
        '''
        return asdict(self)

    def copy(self):
        """
        Returns a deep copy of the BoardLayout instance.
        """
        return BoardLayout(**self.asdict())
  