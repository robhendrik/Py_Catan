import sys  
sys.path.append("../src")
from Py_Catan import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.layers import Input, Embedding, Reshape, Concatenate, Lambda, Dense
from keras.models import Model
from keras.models import load_model
from keras.models import clone_model

def generate_test_data_for_model_with_specific_villages_and_streets(
        file_name_for_logging: str = 'data_for_model_with_specific_villages_and_streets.csv',
        no_of_reps: int = 1000,
        add_random_streets: bool = False
        ) -> None:
    """
    Generates test data for a model that is trained on specific villages and streets.
    The data is saved in a CSV file with the default name 'data_for_model_with_specific_villages_and_streets.csv'.
    This function creates a board structure, initializes a board vector, and generates random data
    for the model. The generated data includes nodes, edges, and hands with specific values.

    Specifically these combinations are used:
        Player one trained to build village on node 0 and street on edge 0.
        Player two trained to build village on node 10 and street on edge 16.
        Player three trained to build village on node 20 and street on edge 25.
        Player four trained to build village on node 30 and street on edge 47
    """
    structure = BoardStructure() 
    brd = Board(structure=structure)
    b_vector = BoardVector(board=brd)
    vf = ValueFunction(preference=pppt.optimized_1_with_0_for_full_score, structure=structure)
    header = b_vector.header()
    edges = b_vector.indices['edges']
    nodes = b_vector.indices['nodes']
    hands = b_vector.indices['hand_for_player']
    values_indices = b_vector.indices['values']
    indices = b_vector.indices
    df = pd.DataFrame(columns=header)
    # === Populate the DataFrame ===
    for _ in range(no_of_reps):
        row_vector = np.zeros(len(header), np.float32)
        if  np.random.choice([0, 1]) == 0:
            row_vector[indices['node_0']] = 1
            if np.random.choice([0, 1]) == 0:
                row_vector[indices['edge_0']] = 1
            else:
                row_vector[indices['edge_5']] = 1

        if np.random.choice([0, 1]) == 0:
            row_vector[indices['node_10']] = 2
            if np.random.choice([0, 1]) == 0:
                row_vector[indices['edge_16']] = 2
            else:
                row_vector[indices['edge_15']] = 2

        if np.random.choice([0, 1]) == 0:
            row_vector[indices['node_20']] = 3
            if np.random.choice([0, 1]) == 0:
                row_vector[indices['edge_25']] = 3
            else:
                row_vector[indices['edge_24']] = 3

        if np.random.choice([0, 1]) == 0:
            row_vector[indices['node_30']] = 4
            if np.random.choice([0, 1]) == 0:
                row_vector[indices['edge_47']] = 4
            else:
                row_vector[indices['edge_48']] = 4

        if add_random_streets:
            for _ in range(5):
                edge = np.random.choice(edges)
                if np.random.choice([0, 1]) == 0:
                    row_vector[edge] = np.random.choice([0,1,2,3])

                node = np.random.choice(nodes)
                if np.random.choice([0, 1]) == 0:
                    row_vector[node] = np.random.choice([0,1,2,3])
        
        for _ in range(5):
            hand = hands[np.random.randint(0, len(hands))]
            if np.random.choice([0, 1]) == 0:
                row_vector[hand] = np.array(b_vector.structure.real_estate_cost[0])
            hand = hands[np.random.randint(0, len(hands))]
            if np.random.choice([0, 1]) == 0:
                row_vector[hand] = np.array(b_vector.structure.real_estate_cost[1])

        values = vf.value_from_vector(row_vector)
        row_vector[values_indices] = values
        df.loc[len(df)] = row_vector

    # === Save the DataFrame to a CSV file ===
    df.to_csv(file_name_for_logging, index=False)
    return

def generate_test_data_for_model_random_streets_villages_towns_hands(
        file_name_for_logging: str = "data_for_model_random_streets_villages_towns_hands.csv",
        no_of_reps = 1000
        ) -> None:
    """
    Generates test data for a model that is trained on random streets, villages, towns, and hands.
    The data is saved in a CSV file with the default name 'data_for_model_random_streets_villages_towns_hands.csv'.
    This function creates a board structure, initializes a board vector, and generates random data
    for the model. The generated data includes nodes, edges, and hands with specific values.

    NOTE: Per rep 20 vectors/rows are generated, so the total number of rows in the DataFrame will be no_of_reps * 20.

    vector contains one of these:
        - 1 to 5 villages with random owners (values 1-4)
        - 1 to 5 towns with random owners (values 5-8)
        - 1 to 5 streets with random owners (values 1-4)
        - 1 to 5 hands with random content (values 0, 1, 2, or 3)

    """
    structure = BoardStructure() 
    brd = Board(structure=structure)
    b_vector = BoardVector(board=brd)
    vf = ValueFunction(preference=pppt.optimized_1_with_0_for_full_score, structure=structure)
    header = b_vector.header()
    values_indices = b_vector.indices['values']
    df = pd.DataFrame(columns=header)
    # === Populate the DataFrame ===
    nodes = b_vector.indices['nodes']
    edges = b_vector.indices['edges']
    hands = b_vector.indices['hand_for_player']
    for n in range(1,6):
        # create vector with 1 to 5 villages with random owners
        for _ in range(no_of_reps):
            vector = np.zeros(b_vector.get_vector_size(),np.float32)
            for _ in range(n):
                node = np.random.randint(0, len(nodes))
                vector[nodes[node]] = np.random.randint(1, 5)
            values = vf.value_from_vector(vector)
            vector[values_indices] = values
            df.loc[len(df)] = vector

        # create vector with 1 to 5 towns with random owners
        for _ in range(no_of_reps):
            vector = np.zeros(b_vector.get_vector_size(),np.float32)
            for _ in range(n):
                node = np.random.randint(0, len(nodes))
                vector[nodes[node]] = np.random.randint(5, 9)
            values = vf.value_from_vector(vector)
            vector[values_indices] = values
            df.loc[len(df)] = vector  

        # create vector with 1 to 5 streets with random owners 
        for _ in range(no_of_reps):
            vector = np.zeros(b_vector.get_vector_size(),np.float32)
            for _ in range(n):
                edge = np.random.randint(0, len(edges))
                vector[edges[edge]] = np.random.randint(1, 5)
            values = vf.value_from_vector(vector)
            vector[values_indices] = values
            df.loc[len(df)] = vector

        # create vector with 1 to 5 hands with random content
        for _ in range(no_of_reps):
            vector = np.zeros(b_vector.get_vector_size(),np.float32)
            for _ in range(n):
                hand = np.random.randint(0, len(hands))
                building = np.random.choice([0,1,2,3])
                vector[hands[hand]] = b_vector.structure.real_estate_cost[building]
            values = vf.value_from_vector(vector)
            vector[values_indices] = values
            df.loc[len(df)] = vector 
    # === Shuffle the DataFrame ===
    df = df.sample(frac=1).reset_index(drop=True)
    # === Save the DataFrame to a CSV file ===
    df.to_csv(file_name_for_logging, index=False)
    return

def split_file_in_test_train_data(
        file_name_for_logging: str = "data_for_model_random_streets_villages_towns_hands.csv",
        no_of_test_vectors: int = 1000,
        file_extension: str = '.csv'
        ) -> tuple:
    """
    Load the csv into a DataFrame, split it into test and train data with attribute 'no_of_test_vectors' determining size
    of the test set. The test set is removed from the DataFrame and saved to a separate file. The remaining DataFrame is saved as the train set.
    The names of the saved files are derived from the original file name by appending '_test' and '_train' respectively and 
    adding the same file extension as the original file.

    Data is shuffled before splitting to ensure randomness in the test and train sets.

    The file names of the train and test sets are returned as a tuple (file_name_train, file_name_test)

    Typical use case: 
    - training_data, testing_data = split_file_in_test_train_data(file_name_for_logging = path_to_data)
                                            
    Args:
        file_name_for_logging (str, optional): _description_. Defaults to "data_for_model_random_streets_villages_towns_hands.csv".
        no_of_test_vectors (int, optional): _description_. Defaults to 1000.
        file_extension (str, optional): _description_. Defaults to '.csv'.

    Returns:
        tuple: A tuple containing the file names of the train and test sets.
    """
    df = pd.read_csv(file_name_for_logging)
    if no_of_test_vectors > len(df):
        raise ValueError(f"no_of_test_vectors ({no_of_test_vectors}) is larger than the number of rows in the DataFrame ({len(df)})")   
    # === Shuffle the DataFrame ===
    df = df.sample(frac=1).reset_index(drop=True)
    # === Split the DataFrame into test and train sets ===
    test_df = df.iloc[:no_of_test_vectors]
    train_df = df.iloc[no_of_test_vectors:] 
    # === Save the DataFrames to CSV files ===
    test_file_name = file_name_for_logging.replace(file_extension, '_test' + file_extension)
    train_file_name = file_name_for_logging.replace(file_extension, '_train' + file_extension)
    # to_csv will overwrite an existing file with the same name
    test_df.to_csv(test_file_name, index=False) 
    train_df.to_csv(train_file_name, index=False)
    return train_file_name, test_file_name

def generate_test_data_from_tournament(
        file_name: str = 'data_from_tournament.csv',
        no_of_games_in_tournament: int = 1000,
        no_of_random_players: int = 3):
    """
    Generates test data from a tournament of players in the Catan game.
    The data is saved in a CSV file with the default name 'data_from_tournament.csv'.
    This function creates a board structure, initializes a board vector, and generates random data
    for the tournament. It sets up players with a mix of random and value function-based strategies.
    """
        
    board_layout = BoardLayout(tile_layout='DSWOSBWWGSGSBGWOGOB')
    structure = BoardStructure(board_layout=board_layout)
    structure.winning_score = 8
    board = Board(structure=structure)
    b_vector = BoardVector(board=board)  
    preference = pppt.optimized_1_with_0_for_full_score

    # === set up players ===
    players = []
    for i in range(no_of_random_players):
        players.append(Player_Random(name=f'Player_{i}', structure=structure))
    for i in range(no_of_random_players, 4):
        players.append(Player_Value_Function_Based(name=f'Player_{i}', structure=structure, preference=preference))

    # === set up tournament ===
    tournament = Tournament()
    tournament.no_games_in_tournament = no_of_games_in_tournament
    tournament.verbose = False
    tournament.logging = True
    tournament.file_name_for_logging = file_name
    player_tournament_results, player_victory_points, rounds_for_this_game = tournament.tournament(structure, players)
    return 

def generate_test_data_from_tournament_with_random_actions(
                                            file_name: str = 'data_from_tournament.csv',
                                            no_of_games_in_tournament: int = 1000,
                                            no_of_random_players: int = 3,
                                            fraction_of_random_actions: float = 0.5):
    """
    Generates test data from a tournament of players in the Catan game.
    The data is saved in a CSV file with the default name 'data_from_tournament.csv'.
    This function creates a board structure, initializes a board vector, and generates random data
    for the tournament. It sets up players with a mix of random and value function-based strategies.
    """
        
    board_layout = BoardLayout(tile_layout='DSWOSBWWGSGSBGWOGOB')
    structure = BoardStructure(board_layout=board_layout)
    structure.winning_score = 8
    board = Board(structure=structure)
    b_vector = BoardVector(board=board)  
    preference = pppt.optimized_1_with_0_for_full_score

    # === set up players ===
    players = []
    for i in range(no_of_random_players):
        players.append(Player_Random(name=f'Player_{i}', structure=structure))
    for i in range(no_of_random_players, 4):
        players.append(Player_Value_Based_With_Randomness(name=f'Player_{i}', 
                                                          structure=structure, 
                                                          preference=preference,
                                                          fraction_of_random_actions=fraction_of_random_actions))

    # === set up tournament ===
    tournament = Tournament()
    tournament.no_games_in_tournament = no_of_games_in_tournament
    tournament.verbose = False
    tournament.logging = True
    tournament.file_name_for_logging = file_name
    player_tournament_results, player_victory_points, rounds_for_this_game = tournament.tournament(structure, players)
    return