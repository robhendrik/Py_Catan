import matplotlib.pyplot as plt
import numpy as np
from Py_Catan.BoardStructure import BoardStructure, BoardLayout
from Py_Catan.Player import Player
from Py_Catan.Board import Board

class PlotCatanBoard(BoardStructure):
    def __init__(self, board: Board = None,
                board_structure: BoardStructure = None,
                layout: BoardLayout = None
                ) -> None   :
        ''' 
        Initializes the PlotCatanBoard with a board, board structure, or layout.
        Args:   
                board (Board): An instance of Board containing the board structure and values.
                board_structure (BoardStructure): An instance of BoardStructure containing the board configuration.
                layout (BoardLayout): An instance of BoardLayout containing the board configuration.
        Returns:
                None
        '''
        if board:
            layout= BoardLayout(board.structure.tile_layout, board.structure.values)
            self.players_from_instance_creation = board.players
            super().__init__(board_layout = layout)   
        elif board_structure:
            layout= BoardLayout(board_structure.tile_layout, board_structure.values)
            self.players_from_instance_creation = None
            super().__init__(board_layout = layout)
        elif layout:
            self.players_from_instance_creation = None
            super().__init__(board_layout=layout)                                           
        else:
            self.players_from_instance_creation = None
            super().__init__()     

    def _plot_board_indicators(self, number: int = 0):
        plt.rcParams["figure.figsize"] = (5,5)
        fig = plt.figure(number)
        ax = fig.add_axes([0.5,0.05,1,1])
        ax.axis('off')
        for tile_number,tile in enumerate(self._tile_coordinates):   
            ax.plot([tile[0]],[tile[1]],marker = r"$ {} $".format(tile_number), color = 'black', markersize = 8) 
        for node_number,node in enumerate(self._node_coordinates):   
            ax.plot([node[0]],[node[1]],marker = 'o',fillstyle = 'full', color = 'black', markersize = 12)                       
            ax.plot([node[0]],[node[1]],marker=r"$ {} $".format(node_number),color = 'white', markersize = 8)
        for edge_number,edge in enumerate(self._edge_coordinates):
            middle = (edge[0]+edge[1])/2
            ax.plot([middle[0]],[middle[1]],marker = 's',fillstyle = 'full', color = 'blue', markersize = 12)            
            ax.plot([middle[0]],[middle[1]],marker=r"$ {} $".format(edge_number),color = 'white', markersize = 8)
        bar_positions = [
            [0.05, 0.75, 0.2, 0.2],  # top-left
            [1.75, 0.75, 0.2, 0.2],  # top-right
            [0.05, 0.05, 0.2, 0.2],  # bottom-left
            [1.75, 0.05, 0.2, 0.2],  # bottom-right
        ]
        for idx,bar_pos in enumerate(bar_positions):
            if idx >= 4:
                break
            ax = plt.gcf().add_axes(bar_positions[idx])
            ax.axis('off')   
        plt.show()     
        return
    
    def plot_board(self, number: int = 1):
        plt.rcParams["figure.figsize"] = (5,5)
        fig = plt.figure(number)
        ax = fig.add_axes([0.5,0.05,1,1])
        ax.axis('off')
        for tile_number,tile in enumerate(self._tile_coordinates):   
            ax.plot([tile[0]],[tile[1]],marker = 'H', color = 'lightgrey', markersize = 45) 
            text = self.tile_layout[tile_number] + '  /  ' + str(self.values[tile_number])
            ax.plot([tile[0]],[tile[1]],marker = r"$ {} $".format(text), color = 'black', markersize = 15) 
        for node_number,node in enumerate(self._node_coordinates):   
            ax.plot([node[0]],[node[1]],marker = 'o',fillstyle = 'full', color = 'grey', markersize = 8)                       
        for edge_number,edge in enumerate(self._edge_coordinates):
            start = edge[0]- 0.4 *(edge[0]-edge[1])
            end = edge[0]- 0.6 *(edge[0]-edge[1])
            ax.plot([start[0],end[0]],[start[1],end[1]], color = 'grey',  linewidth=5.0)  

        bar_positions = [
            [0.05, 0.75, 0.2, 0.2],  # top-left
            [1.75, 0.75, 0.2, 0.2],  # top-right
            [0.05, 0.05, 0.2, 0.2],  # bottom-left
            [1.75, 0.05, 0.2, 0.2],  # bottom-right
        ]
        for idx,bar_pos in enumerate(bar_positions):
            if idx >= 4:
                break
            ax = plt.gcf().add_axes(bar_positions[idx])
            ax.axis('off')
        #plt.show()
        return fig

    def plot_board_positions(self,players: list = None, number: int = 2, fig: plt.figure = None):
        if not players:
            if self.players_from_instance_creation:
                players = self.players_from_instance_creation
            else:
                raise Exception("No players provided and no players from instance creation available.")
            
        if not fig:
            plt.rcParams["figure.figsize"] = (5,5)
            fig = self.plot_board(number = number)
            ax = fig.get_axes()[0]
        else:
            ax = fig.get_axes()[0]
        for p,color in zip(players,self.plot_colors_players):
            for node_number,node in enumerate(self._node_coordinates):   
                if p.towns[node_number]:
                    ax.plot([node[0]],[node[1]],marker = '*',fillstyle = 'full', color = color, markersize = 20,zorder =100)      
                if p.villages[node_number]:
                    ax.plot([node[0]],[node[1]],marker = 'o',fillstyle = 'full', color = color, markersize = 15,zorder =100)       
            for edge_number,edge in enumerate(self._edge_coordinates):
                if p.streets[edge_number]:
                    start = edge[0]- 0.25 *(edge[0]-edge[1])
                    end = edge[0]- 0.75 *(edge[0]-edge[1])
                    ax.plot([start[0],end[0]],[start[1],end[1]], color = color,  linewidth=6.0,zorder =100) 
        # Add per player a bar graph of p.hand in the 4 corners for up to 4 players
        hand_labels = self.plot_labels_for_resources
        bar_positions = [
            [0.05, 0.75, 0.2, 0.2],  # top-left
            [1.75, 0.75, 0.2, 0.2],  # top-right
            [0.05, 0.05, 0.2, 0.2],  # bottom-left
            [1.75, 0.05, 0.2, 0.2],  # bottom-right
        ]
        for idx, (p, color) in enumerate(zip(players, self.plot_colors_players)):
            if idx >= 4:
                break
            if len(fig.axes) <= idx + 1:
                ax = fig.add_axes(bar_positions[idx])
            else:
                ax = fig.axes[idx + 1]
            ax.bar(hand_labels,  p.hand, color=color)
            ax.set_title(p.name, fontsize=8)
            ax.set_xticks(np.arange(len(hand_labels)))
            ax.set_xticklabels(hand_labels, rotation=90, fontsize=7)
            ax.set_yticks([])
            ax.set_ylim(0, self.plot_max_card_in_hand_per_type)
            ax.tick_params(axis='x', which='both', length=0, labelsize=6)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.hlines( y=list(range(1,self.plot_max_card_in_hand_per_type)),
                        xmin=-1,xmax=len(hand_labels),
                        colors=['w'] *self.plot_max_card_in_hand_per_type,
                        linestyles=['-']*self.plot_max_card_in_hand_per_type)
        plt.show()
        return   