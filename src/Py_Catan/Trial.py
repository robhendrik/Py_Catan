import sys  
sys.path.append("../src")
import numpy as np
import matplotlib.pyplot as plt

from Py_Catan.Board import Board
from Py_Catan.BoardStructure import BoardStructure, BoardLayout
from Py_Catan.Player import Player
from Py_Catan.Preferences import PlayerPreferences
from Py_Catan.PlayerValueFunctionBased import Player_Value_Function_Based
from Py_Catan.PlayerRandom import Player_Random
from Py_Catan.PlayerPassive import Player_Passive
from Py_Catan.PlotBoard import PlotCatanBoard
from Py_Catan.Tournament import Tournament
import Py_Catan.Player_Preference_Types as pppt

class Trial:
    def __init__(self):
        self.number_of_tournaments = 100
        self.number_of_games_in_tournament = 24
        board_layout = BoardLayout(tile_layout='DSWOSBWWGSGSBGWOGOB')
        self.structure = BoardStructure(board_layout=board_layout)
        self.structure.winning_score = 8
        self.mediocre_players = [
            Player_Value_Function_Based(name = 'mediocre_1',structure=self.structure,preference=pppt.mediocre_1),
            Player_Value_Function_Based(name = 'mediocre_2',structure=self.structure,preference=pppt.mediocre_2),
            Player_Value_Function_Based(name = 'mediocre_1a',structure=self.structure,preference=pppt.mediocre_1)
        ]
        self.strong_players = [
            Player_Value_Function_Based(name = 'strong_1',structure=self.structure,preference=pppt.strong_1),
            Player_Value_Function_Based(name = 'strong_2',structure=self.structure,preference=pppt.strong_2),
            Player_Value_Function_Based(name = 'strong_1a',structure=self.structure,preference=pppt.strong_1)
        ]

        self.optimized_players = [
            Player_Value_Function_Based(name = 'optimized_1',structure=self.structure,preference=pppt.optimized_1),
            Player_Value_Function_Based(name = 'optimized_2',structure=self.structure,preference=pppt.optimized_2),
            Player_Value_Function_Based(name = 'optimized_1a',structure=self.structure,preference=pppt.optimized_1)
        ]

        self.earlier_results_1 = [{'passive': {'victory_points': (np.float64(240.0), np.float64(0.0)), 'results': (np.float64(81.0), np.float64(5.347896782848375)), 'rounds': (np.float64(51.0), np.float64(0.0)), 'name': 'Player_0'}, 'random': {'victory_points': (np.float64(102.50833333333333), np.float64(11.695883060100916)), 'results': (np.float64(112.7), np.float64(7.430343195303969)), 'rounds': (np.float64(41.8), np.float64(10.4)), 'name': 'Player_0'}, 'mediocre': {'victory_points': (np.float64(80.15), np.float64(14.74485786073684)), 'results': (np.float64(61.5), np.float64(6.422616289332565)), 'rounds': (np.float64(27.4), np.float64(9.3936148526539)), 'name': 'Player_0'}, 'strong': {'victory_points': (np.float64(9.083333333333334), np.float64(5.900211860602973)), 'results': (np.float64(67.5), np.float64(5.678908345800274)), 'rounds': (np.float64(22.2), np.float64(5.861740355901138)), 'name': 'Player_0'}}, {'passive': {'victory_points': (np.float64(240.0), np.float64(0.0)), 'results': (np.float64(48.0), np.float64(0.0)), 'rounds': (np.float64(51.0), np.float64(0.0)), 'name': 'Player_1'}, 'random': {'victory_points': (np.float64(31.458333333333332), np.float64(5.735277044553104)), 'results': (np.float64(54.3), np.float64(4.001249804748511)), 'rounds': (np.float64(45.7), np.float64(9.089004345911603)), 'name': 'Player_1'}, 'mediocre': {'victory_points': (np.float64(124.53333333333335), np.float64(7.395118510056334)), 'results': (np.float64(94.7), np.float64(3.925557285278104)), 'rounds': (np.float64(46.4), np.float64(7.735631842325486)), 'name': 'Player_1'}, 'strong': {'victory_points': (np.float64(2.4666666666666672), np.float64(1.0349449797506685)), 'results': (np.float64(48.0), np.float64(0.0)), 'rounds': (np.float64(19.3), np.float64(5.64003546088143)), 'name': 'Player_1'}}, {'passive': {'victory_points': (np.float64(240.0), np.float64(0.0)), 'results': (np.float64(188.2), np.float64(1.9899748742132397)), 'rounds': (np.float64(26.7), np.float64(13.51332675546625)), 'name': 'Player_2'}, 'random': {'victory_points': (np.float64(212.95), np.float64(7.607397715382047)), 'results': (np.float64(181.5), np.float64(4.295346318982906)), 'rounds': (np.float64(22.8), np.float64(12.310970717209914)), 'name': 'Player_2'}, 'mediocre': {'victory_points': (np.float64(211.25), np.float64(7.846177413237608)), 'results': (np.float64(162.5), np.float64(5.220153254455275)), 'rounds': (np.float64(30.9), np.float64(15.604166110369372)), 'name': 'Player_2'}, 'strong': {'victory_points': (np.float64(110.48333333333332), np.float64(12.611822055339799)), 'results': (np.float64(144.8), np.float64(5.0556898639058145)), 'rounds': (np.float64(21.1), np.float64(5.008991914547277)), 'name': 'Player_2'}}, {'passive': {'victory_points': (np.float64(240.0), np.float64(0.0)), 'results': (np.float64(191.7), np.float64(0.6403124237432849)), 'rounds': (np.float64(22.2), np.float64(5.582114294781145)), 'name': 'Player_3'}, 'random': {'victory_points': (np.float64(237.7), np.float64(2.9342801502242417)), 'results': (np.float64(192.5), np.float64(1.746424919657298)), 'rounds': (np.float64(14.8), np.float64(3.8157568056677826)), 'name': 'Player_3'}, 'mediocre': {'victory_points': (np.float64(239.0), np.float64(2.0)), 'results': (np.float64(193.1), np.float64(1.57797338380595)), 'rounds': (np.float64(17.8), np.float64(4.354308211415448)), 'name': 'Player_3'}, 'strong': {'victory_points': (np.float64(157.15), np.float64(14.951486362380312)), 'results': (np.float64(160.9), np.float64(5.769748694700662)), 'rounds': (np.float64(18.3), np.float64(5.020956084253276)), 'name': 'Player_3'}}]
        self.earlier_results_2_30_tournaments_20_games = [{'passive': {'victory_points': (np.float64(240.0), np.float64(0.0)), 'results': (np.float64(84.96666666666667), np.float64(7.27774385064187)), 'rounds': (np.float64(51.0), np.float64(0.0)), 'name': 'Player_0'}, 'random': {'victory_points': (np.float64(98.01666666666667), np.float64(17.41753586347838)), 'results': (np.float64(113.0), np.float64(9.44810386620864)), 'rounds': (np.float64(37.46666666666667), np.float64(11.104753736826204)), 'name': 'Player_0'}, 'mediocre': {'victory_points': (np.float64(85.23055555555557), np.float64(13.798931173942774)), 'results': (np.float64(65.73333333333333), np.float64(8.131967098364878)), 'rounds': (np.float64(36.43333333333333), np.float64(12.69562479316748)), 'name': 'Player_0'}, 'strong': {'victory_points': (np.float64(7.622222222222223), np.float64(6.161549428351985)), 'results': (np.float64(64.46666666666667), np.float64(5.090732320163334)), 'rounds': (np.float64(20.6), np.float64(4.687572221665851)), 'name': 'Player_0'}}, {'passive': {'victory_points': (np.float64(240.0), np.float64(0.0)), 'results': (np.float64(48.0), np.float64(0.0)), 'rounds': (np.float64(51.0), np.float64(0.0)), 'name': 'Player_1'}, 'random': {'victory_points': (np.float64(31.869444444444444), np.float64(10.305331309330223)), 'results': (np.float64(55.06666666666667), np.float64(5.189305241445032)), 'rounds': (np.float64(43.4), np.float64(11.388883468833399)), 'name': 'Player_1'}, 'mediocre': {'victory_points': (np.float64(124.55555555555556), np.float64(7.933442265047339)), 'results': (np.float64(95.1), np.float64(5.061949558552844)), 'rounds': (np.float64(43.93333333333333), np.float64(10.243480311343838)), 'name': 'Player_1'}, 'strong': {'victory_points': (np.float64(2.144444444444445), np.float64(1.514579353247611)), 'results': (np.float64(48.0), np.float64(0.0)), 'rounds': (np.float64(22.533333333333335), np.float64(8.689968674026133)), 'name': 'Player_1'}}, {'passive': {'victory_points': (np.float64(240.0), np.float64(0.0)), 'results': (np.float64(189.16666666666666), np.float64(1.9677962857527254)), 'rounds': (np.float64(26.733333333333334), np.float64(13.251247320745152)), 'name': 'Player_2'}, 'random': {'victory_points': (np.float64(210.49444444444447), np.float64(10.190180761429959)), 'results': (np.float64(179.53333333333333), np.float64(5.277204647243547)), 'rounds': (np.float64(28.5), np.float64(14.497700967164874)), 'name': 'Player_2'}, 'mediocre': {'victory_points': (np.float64(217.74444444444447), np.float64(8.740681104630205)), 'results': (np.float64(162.93333333333334), np.float64(6.884443009826979)), 'rounds': (np.float64(32.766666666666666), np.float64(13.97541492129025)), 'name': 'Player_2'}, 'strong': {'victory_points': (np.float64(108.90000000000002), np.float64(13.660947312740849)), 'results': (np.float64(146.76666666666668), np.float64(6.95549343245243)), 'rounds': (np.float64(20.366666666666667), np.float64(3.5635500401830886)), 'name': 'Player_2'}}, {'passive': {'victory_points': (np.float64(240.0), np.float64(0.0)), 'results': (np.float64(191.46666666666667), np.float64(1.2036980056845192)), 'rounds': (np.float64(26.7), np.float64(9.081666513733406)), 'name': 'Player_3'}, 'random': {'victory_points': (np.float64(238.36666666666667), np.float64(3.3214789209360074)), 'results': (np.float64(193.16666666666666), np.float64(1.9846634195472261)), 'rounds': (np.float64(15.266666666666667), np.float64(4.396463225012679)), 'name': 'Player_3'}, 'mediocre': {'victory_points': (np.float64(239.16666666666666), np.float64(2.1730674684008826)), 'results': (np.float64(194.33333333333334), np.float64(1.6996731711975948)), 'rounds': (np.float64(18.233333333333334), np.float64(5.187699126030944)), 'name': 'Player_3'}, 'strong': {'victory_points': (np.float64(151.4), np.float64(16.58293411021414)), 'results': (np.float64(158.63333333333333), np.float64(8.388417941158842)), 'rounds': (np.float64(18.233333333333334), np.float64(5.264239947249957)), 'name': 'Player_3'}}]
        return

    def trial_against_passive_player(self,player)-> dict:
        """ 
        Run a trial against a passive player to see how the value function based player performs.
        """
        vic_pts = []
        results = []
        rounds = []
 
        for _ in range(self.number_of_tournaments):
            players = [player]
            for name in ['B','C','D']:
                players.append(Player_Passive(name = name, structure = self.structure ))
            tournament = Tournament()
            tournament.no_games_in_tournament = self.number_of_games_in_tournament
            tournament.verbose = False
            # Run the tournament
            player_tournament_results, player_victory_points, rounds_for_this_game = tournament.tournament(
                self.structure, players)
            vic_pts.append(player_victory_points[0])
            results.append(player_tournament_results[0])
            rounds.append(rounds_for_this_game[0]) 
        result = {
            'victory_points': (np.mean(np.array(vic_pts)),np.std(np.array(vic_pts))),
            'results': (np.mean(np.array(results)), np.std(np.array(results))),
            'rounds': (np.mean(np.array(rounds)), np.std(np.array(rounds))),
            'name': player.name
        }
        return result
    
    def trial_against_random_player(self,player)-> dict:
        vic_pts = []
        results = []
        rounds = []

        for _ in range(self.number_of_tournaments):
            players = [player]
            for name in ['B','C','D']:
                players.append(Player_Random(name = name, structure = self.structure ))
                players[-1].threshold_for_accepting_trade = 0.5
                players[-1].max_actions_in_round = 5
            tournament = Tournament()
            tournament.no_games_in_tournament = self.number_of_games_in_tournament
            tournament.verbose = False
            # Run the tournament
            player_tournament_results, player_victory_points, rounds_for_this_game = tournament.tournament(
                self.structure, players)
            vic_pts.append(player_victory_points[0])
            results.append(player_tournament_results[0])
            rounds.append(rounds_for_this_game[0])  
        result = {
            'victory_points': (np.mean(np.array(vic_pts)),np.std(np.array(vic_pts))),
            'results': (np.mean(np.array(results)), np.std(np.array(results))),
            'rounds': (np.mean(np.array(rounds)), np.std(np.array(rounds))),
            'name': player.name
        }
        return result

    def trial_against_mediocre_players(self,player)-> dict:
        vic_pts = []
        results = []
        rounds = []

        for _ in range(self.number_of_tournaments):
            players = [player]
            for default_player in self.mediocre_players:
                players.append(default_player)
            tournament = Tournament()
            tournament.no_games_in_tournament = self.number_of_games_in_tournament
            tournament.verbose = False
            # Run the tournament
            player_tournament_results, player_victory_points, rounds_for_this_game = tournament.tournament(
                self.structure, players)
            vic_pts.append(player_victory_points[0])
            results.append(player_tournament_results[0])
            rounds.append(rounds_for_this_game[0]) 
        result = {
            'victory_points': (np.mean(np.array(vic_pts)),np.std(np.array(vic_pts))),
            'results': (np.mean(np.array(results)), np.std(np.array(results))),
            'rounds': (np.mean(np.array(rounds)), np.std(np.array(rounds))),
            'name': player.name
        }
        return result
    
    def trial_against_strong_players(self,player)-> dict:
        vic_pts = []
        results = []
        rounds = []
        for _ in range(self.number_of_tournaments):
            players = [player]
            for default_player in self.strong_players:
                players.append(default_player)
            tournament = Tournament()
            tournament.no_games_in_tournament = self.number_of_games_in_tournament
            tournament.verbose = False
            # Run the tournament
            player_tournament_results, player_victory_points, rounds_for_this_game = tournament.tournament(
                self.structure, players)
            vic_pts.append(player_victory_points[0])
            results.append(player_tournament_results[0])
            rounds.append(rounds_for_this_game[0]) 
        result = {
            'victory_points': (np.mean(np.array(vic_pts)),np.std(np.array(vic_pts))),
            'results': (np.mean(np.array(results)), np.std(np.array(results))),
            'rounds': (np.mean(np.array(rounds)), np.std(np.array(rounds))),
            'name': player.name
        }
        return result
    
    def trial_against_optimized_players(self,player)-> dict:
        vic_pts = []
        results = []
        rounds = []
        for _ in range(self.number_of_tournaments):
            players = [player]
            for default_player in self.optimized_players:
                players.append(default_player)
            tournament = Tournament()
            tournament.no_games_in_tournament = self.number_of_games_in_tournament
            tournament.verbose = False
            # Run the tournament
            player_tournament_results, player_victory_points, rounds_for_this_game = tournament.tournament(
                self.structure, players)
            vic_pts.append(player_victory_points[0])
            results.append(player_tournament_results[0])
            rounds.append(rounds_for_this_game[0]) 
        result = {
            'victory_points': (np.mean(np.array(vic_pts)),np.std(np.array(vic_pts))),
            'results': (np.mean(np.array(results)), np.std(np.array(results))),
            'rounds': (np.mean(np.array(rounds)), np.std(np.array(rounds))),
            'name': player.name
        }
        return result
    
    def run_trials(self,player):
        passive_player_result = self.trial_against_passive_player(player)
        mediocre_player_result = self.trial_against_mediocre_players(player)
        strong_player_result = self.trial_against_strong_players(player)
        random_player_result = self.trial_against_random_player(player)
      
        return {
            'passive': passive_player_result,
            'random': random_player_result,
            'mediocre': mediocre_player_result,
            'strong': strong_player_result,
        }
    
    def run_trials_for_all_players(self,players):
        results = []
        for player in players:
            result = self.run_trials(player)
            results.append(result)
        return results
    
    def visualize_results(self,results):
        """
        Visualizes the results of the trials.
        """
        for score_type in ['victory_points','results','rounds']:
            # Extract average and std victory points for each player and trial type
            trial_types = ['passive', 'random', 'mediocre', 'strong']
            p_means = [None]*len(results)
            p_stds = [None]*len(results)
            p_name = [None]*len(results)
            for n in range(len(results)):
                p_means[n] = [results[n][t][score_type][0] for t in trial_types]
                p_stds[n] = [results[n][t][score_type][1] for t in trial_types]
                p_name[n] = results[n][trial_types[0]]['name']
        

            x = np.arange(len(trial_types))  # the label locations
            width = 1/(2*len(results))  # the width of the bars

            fig, ax = plt.subplots(figsize=(8, 6))
            for n in range(len(results)):
                    rects1 = ax.bar(x - 1/4 + n*width, p_means[n], width, yerr=p_stds[n], label=p_name[n], capsize=5)

            ax.set_ylabel(score_type)
            ax.set_title('Player Scores Across Trials: '+score_type)
            ax.set_xticks(x)
            ax.set_xticklabels(trial_types)
            ax.legend()

            plt.tight_layout()
            plt.show()  