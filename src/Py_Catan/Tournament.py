import sys  
sys.path.append("../src")
import numpy as np

from Py_Catan.Board import Board
from Py_Catan.BoardStructure import BoardStructure, BoardLayout
from Py_Catan.Player import Player
from Py_Catan.Preferences import PlayerPreferences
from Py_Catan.PlayerValueFunctionBased import Player_Value_Function_Based
from Py_Catan.PlotBoard import PlotCatanBoard
import Py_Catan.Player_Preference_Types as pppt

class Tournament:
    def __init__(self):
        '''
        Initialize the tournament with players and board structure.
        The tournament always has 4 players.
        '''
        self.no_games_in_tournament = 16
        self.score_table_for_ranking_per_game = [10, 5, 2, 0]
        self.max_rounds_per_game = 50
        self.verbose = True
        self.logging = False
        self.list_of_orders = self._create_list_of_orders()
        self.list_of_reversed_orders = self._create_list_of_reversed_orders()
        random_indicator = np.random.randint(0,1000)
        self.file_name_for_logging = f"game_logging_{random_indicator}.txt"

    def _create_list_of_orders(self) -> list:
        '''
        Create a list of orders for the players. It is based on 4 players. 
        All permutations will be in the list.

        returns:
            list_of_orders: list of lists, where each list contains the indices of the players in the order they will play.
        '''
        list_of_orders = [[i] for i in range(4)]
        for _ in range(3):
            new_list = []
            for l in list_of_orders:
                for i in range(4):
                    if i not in l:
                        new_list.append(l+[i])
            list_of_orders = new_list
        return list_of_orders
    
    def _create_list_of_reversed_orders(self) -> list:
        '''
        Create a list of reversed orders for the players.
        The list of reversed orders is a list of lists, where each list contains the indices of the players in the order they will play.
        The order is determined by the game number modulo the length of the list of orders.

        NOTE: We first need to create self.list_of_orders before we can create self.list_of_reversed_orders.

        returns:
            list_of_reversed_orders: list of lists, where each list is used to retrieve the original order.
        '''
        list_of_reversed_orders = []
        for order in self.list_of_orders:
            reverse_order = [0]*len(order)
            for i,j in enumerate(order):
                reverse_order[j] = i
            list_of_reversed_orders.append(reverse_order)
        return list_of_reversed_orders


    def _order_elements(self,game_number: int,elements:list,reverse: bool = False) -> list:
        if not reverse:
            order = self.list_of_orders[game_number % len(self.list_of_orders)]
            ordered_elements = [elements[i] for i in order]
            return ordered_elements
        else:
            reverse_order = self.list_of_reversed_orders[game_number % len(self.list_of_reversed_orders)]
            ordered_elements = [elements[i] for i in reverse_order]     
            return ordered_elements
        
    def tournament(self,board_structure, players) -> tuple:
        '''
        Run a tournament with the given board structure and players.
        Returns the results of the tournament, including total results, victory points, and rounds played.
        The players are expected to be a list of Player objects.
        The board_structure is expected to be a BoardStructure object.
        The players are expected to be a list of Player objects.
        The tournament will run for a fixed number of games, with players playing in different orders.
        The results will be calculated based on the scores of the players.
        The function will return the total results, victory points, and rounds played for each player.
        The players are expected to be a list of Player objects.
        The board_structure is expected to be a BoardStructure object.
        '''
        if len(players) != 4:
            raise ValueError("The tournament requires exactly 4 players.")
        verbose = self.verbose
        player_tournament_results = np.zeros(len(players),np.int16)
        player_victory_points = np.zeros(len(players),np.float64)
        rounds_for_this_tournament = np.zeros(len(players),np.int16)

        if self.logging:
            with open(self.file_name_for_logging , "w") as f:
                brd= Board(structure=board_structure, players=players)
                f.write(",".join(brd.logging_header()) + "\n")
                del brd
            self.game_records = dict([])

        for game_number in range(self.no_games_in_tournament):
            players_for_this_game = self._order_elements(game_number, [p.copy() for p in players],reverse = False)
            for player in players_for_this_game:
                player._players_in_this_game = players_for_this_game
   
            results, rounds = self.play_game(board_structure=board_structure,players=players_for_this_game)
            results_for_this_game = np.array(self._order_elements(game_number, results, reverse=True))
            rounds_for_this_game = np.array(rounds)
            points_for_this_game = np.array(self.calculate_points(results_for_this_game))
            player_names = self._order_elements(game_number, [p.name for p in players_for_this_game], reverse=True)

            if verbose:
                print('\nResults for this game:')
                print('Player\tResults\tPoints\tRounds')
                for i, p in enumerate(players_for_this_game):
                    print(f"{player_names[i]}\t{results_for_this_game[i]}\t{points_for_this_game[i]}\t{rounds_for_this_game[i]}")

            player_tournament_results += results_for_this_game
            player_victory_points += points_for_this_game
            rounds_for_this_tournament += rounds_for_this_game
            # === Log the game records if logging is enabled ===
            if self.logging:
                ranking = self.game_ranking(results)
                ranking_indices = board_structure.vector_indices['ranks']
                turn_before_end_index = board_structure.vector_indices['turns']
                with open(self.file_name_for_logging , "a") as f:
                    for round_position,vector in self.game_records.items():
                        # rounds is the total number of rounds in the game
                        round,position = round_position
                        vector[turn_before_end_index] = [rounds[0]-round-1]
                        vector[ranking_indices] = ranking
                        np.savetxt(f, vector[None,:].astype(np.float64), delimiter=', ', fmt="%.4f")    
                self.game_records = dict([])

        if verbose:
            print('\nFinal Tournament Results:')
            print('Player\tTotal Results\tTotal Points\tTotal Rounds')
            for i, p in enumerate(players):
                print(f"{p.name}\t{player_tournament_results[i]}\t{player_victory_points[i]}\t{rounds_for_this_tournament[i]}")

        return player_tournament_results, player_victory_points, rounds_for_this_game

    def calculate_points(self,results):
        '''
        Calculate the points for each player based on their results.
        The points are calculated based on the score table for ranking per game.    
        if multiple players have the same score, they will receive the average of the scores for those positions.
        '''
        score_table = self.score_table_for_ranking_per_game
        temp_table = score_table.copy()
        temp_results = results.copy()
        points = np.zeros(len(results),np.float64)
        while max(temp_results) > 0:              
            max_value = max(temp_results)
            indices = [i for i, j in enumerate(temp_results) if j == max_value]
            score = sum(temp_table[:len(indices)])/len(indices)
            for i in indices:
                points[i] = score
                temp_results[i] = -1000
            temp_table = temp_table[len(indices):]
        return points
    
    def game_ranking(self, results):
        temp_results = results.copy()
        ranking = [0] * len(results)
        rank = 1
        while max(temp_results) >= 0:              
            max_value = max(temp_results)
            indices = [i for i, j in enumerate(temp_results) if j == max_value]
            for i in indices:
                ranking[i] = rank
                temp_results[i] = -1000
            rank += len(indices)
        return ranking
    
    def create_log_from_board(self,board):
        d = {p.name: {'hand':p.hand, 'streets':p.streets, 'villages':p.villages, 'towns':p.towns}  
        for p in board.players}
        return d
    
    def play_game(self,board_structure: BoardStructure,players: list, draw_board: bool =False)->tuple:
        '''
        Play a game with the given board structure and players.
        Returns the scores of the players and the rounds played
        '''
        brd = Board(structure = board_structure,players=players)
        brd._update_board_for_players()
        # Set up the board
        self.setup_board(brd)
        if draw_board:
            draw = PlotCatanBoard(board = brd)
            draw.plot_board_positions()   
        # Play rounds until one player reaches the winning score or the maximum number of rounds is reached
        round = 0
        while all([p.calculate_score() < brd.structure.winning_score for p in brd.players]) and (round<=self.max_rounds_per_game):
            self.play_round(brd, round)
            round += 1
        # Document the final state if needed
        if draw_board:
            draw = PlotCatanBoard(board = brd)
            draw.plot_board_positions()   

        return ([p.calculate_score() for p in brd.players],[round]*len(brd.players))

    def setup_board(self,brd):
        '''    
        Setup the board with players and initial buildings.
        '''
        for p in brd.players + brd.players[::-1]:
            actions = p.player_setup(brd)
            brd.execute_player_action(p, actions[0])
            brd.execute_player_action(p, actions[1])
        return

    def play_round(self,brd, round: int = 0):
        '''
        Play a round for all players in the board.
        '''
        for (position,player) in enumerate(brd.players):
            self.player_turn(brd, player)
            if self.logging:
                self.game_records.update({(round,position):brd.create_board_vector()})
        return  

    
        
    def player_turn(self,brd,player):
        '''
        Play a turn for the given player.
        '''
        brd.throw_dice()
        rejected_trades_for_this_round = set([])
        rejected_trades_for_this_round_for_specific_player = set([])
        while True:
            # execute actions until the player has no more actions left or no more actions can be executed
            if not self.player_action_in_turn(brd,
                                              player,
                                              rejected_trades_for_this_round,
                                              rejected_trades_for_this_round_for_specific_player):
                return

    def player_action_in_turn(self,
                              brd,
                              player,
                              rejected_trades_for_this_round,
                              rejected_trades_for_this_round_for_specific_player: set = set([])):
            player.update_build_options()
            best_action = player.find_best_action(  rejected_trades_for_this_round,
                                                    rejected_trades_for_this_round_for_specific_player)
            if best_action[0] is None:
                return False
            brd.execute_player_action(player, 
                                      best_action,
                                      rejected_trades_for_this_round,
                                      rejected_trades_for_this_round_for_specific_player)
            player._actions_in_round += 1
            return True
            
    


    