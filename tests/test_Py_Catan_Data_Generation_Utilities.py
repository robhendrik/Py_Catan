import sys  
sys.path.append("../src")
sys.path.append("./src")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.layers import Input, Embedding, Reshape, Concatenate, Lambda, Dense
from keras.models import Model
from keras.models import load_model
from keras.models import clone_model
from Py_Catan.Player_Model_Types import PlayerModelTypes
from Py_Catan.Player_Model_Types import model_trained_on_streets_villages_towns_b_cards,model_trained_on_limited_tournament_data
from Py_Catan.Player_Model_Types import model_trained_on_specific_villages_and_streets,blank_model
from Py_Catan.DataGenerationUtils import generate_test_data_for_model_with_specific_villages_and_streets
from Py_Catan.DataGenerationUtils import generate_test_data_for_model_random_streets_villages_towns_hands
from Py_Catan.DataGenerationUtils import split_file_in_test_train_data
from Py_Catan.DataGenerationUtils import generate_test_data_from_tournament
from Py_Catan.BoardVector import BoardVector

def test_generate_test_data_for_model_with_specific_villages_and_streets():
    """
    Test function for generate_test_data_for_model_with_specific_villages_and_streets.
    It generates a test file and checks if the DataFrame is not empty.
    """
    file_name = './tests/test_models/test_data_for_model_with_specific_villages_and_streets.csv'
    no_of_reps = 1000
    generate_test_data_for_model_with_specific_villages_and_streets(file_name, no_of_reps)
    
    df = pd.read_csv(file_name)
    assert not df.empty, "DataFrame is empty after generating test data"
    assert len(df) == no_of_reps, f"DataFrame size mismatch: {len(df)} != {no_of_reps}"
    b_vector = BoardVector()
    indices = b_vector.get_indices()
    for _, row in df.iterrows():
        # assert that if villages are present on 0,10,20 or 30 then street are also present on 0 or 5,16 or 15,24 or 25 or 47 or 48
        if row.iloc[indices['node_0']] > 0:
            assert row.iloc[indices['edge_0']] > 0 or row.iloc[indices['edge_5']] > 0, \
                f"Row {row} has village on node 0 but no street on edge 0 or 5: {row.iloc[indices['edge_0']]} {row.iloc[indices['edge_5']]}"  
        if row.iloc[indices['node_10']] > 0:
            assert row.iloc[indices['edge_16']] > 0 or row.iloc[indices['edge_15']] > 0, \
                f"Row {row} has village on node 10 but no street on edge 16 or 15: {row.iloc['edge_16']} {row.iloc['edge_15']}"
        if row.iloc[indices['node_20']] > 0:
            assert row.iloc[indices['edge_25']] > 0 or row.iloc[indices['edge_24']] > 0, \
                f"Row {row} has village on node 20 but no street on edge 25 or 24: {row.iloc[indices['edge_25']]} {row.iloc[indices['edge_24']]}"
        if row.iloc[indices['node_30']] > 0:
            assert row.iloc[indices['edge_47']] > 0 or row.iloc[indices['edge_48']] > 0, \
                f"Row {row} has village on node 30 but no street on edge 47 or 48: {row.iloc[indices['edge_47']]} {row.iloc[indices['edge_48']]}"
    return

def test_generate_test_data_for_model_random_streets_villages_towns_hands():
    """
    Test function for generate_test_data_for_model_random_streets_villages_towns_hands.
    It generates a test file and checks if the DataFrame is not empty.
    """
    file_name = './tests/test_models/test_data_for_model_random_streets_villages_towns_hands.csv'
    no_of_reps = 1000
    generate_test_data_for_model_random_streets_villages_towns_hands(file_name, no_of_reps)
    b_vector = BoardVector()
    indices = b_vector.get_indices()
    df = pd.read_csv(file_name)
    assert not df.empty, "DataFrame is empty after generating test data"
    assert len(df) == no_of_reps * 20, f"DataFrame size mismatch: {len(df)} != {no_of_reps * 20}"
    for _, row in df.iterrows():
        num_streets = sum([row.iloc[i] > 0 for i in indices['edges']])
        num_villages = sum([1 <= row.iloc[i] <= 4 for i in indices['nodes']])
        num_towns = sum([5 <= row.iloc[i] <= 8 for i in indices['nodes']])
        num_cards = sum([row.iloc[i] for i in indices['hands']])
        # test at least one of each type is present
        assert num_streets > 0 or num_villages > 0 or num_towns > 0 or num_cards > 0, \
            f"Row {row} does not contain any streets, villages, towns, or cards"
        # test not more than one type is present
        assert (num_streets > 0) + (num_villages > 0) + (num_towns > 0) + (num_cards > 0) <= 1, \
            f"Row {row} contains more than one type of data: streets={num_streets}, villages={num_villages}, towns={num_towns}, cards={num_cards}"
        # test not more than 5 of each type is present
        assert num_streets <= 5, f"Row {row} contains more than 5 streets: {num_streets}"
        assert num_villages <= 5, f"Row {row} contains more than 5 villages: {num_villages}"
        assert num_towns <= 5, f"Row {row} contains more than 5 towns: {num_towns}"

def test_split_file_in_test_train_data():
    """
    Test function for split_file_in_test_train_data.
    It generates a test file, splits it into test and train data, and checks the sizes of the resulting DataFrames.
    """
    file_name = './tests/test_models/test_data_for_split_testing.csv'
    no_of_test_vectors = 100
    generate_test_data_for_model_random_streets_villages_towns_hands(file_name, no_of_reps=500)
    split_file_in_test_train_data(file_name, no_of_test_vectors)
    df = pd.read_csv(file_name)
    assert not df.empty, "DataFrame is empty after generating test data"
    assert len(df) == 500*20, f"DataFrame size mismatch: {len(df)} != 500*20"
    
    df = pd.read_csv(file_name.replace('.csv', '_train.csv'))
    assert len(df) == 500*20 - no_of_test_vectors, f"Train DataFrame size mismatch: {len(df)} != {500*20 - no_of_test_vectors}"
    
    df = pd.read_csv(file_name.replace('.csv', '_test.csv'))
    assert len(df) == no_of_test_vectors, f"Test DataFrame size mismatch: {len(df)} != {no_of_test_vectors}"

def test_generate_test_data_from_tournament():
    """
    Test function for generate_test_data_from_tournament.
    It generates a test file and checks if the DataFrame is not empty.
    """
    file_name = './tests/test_models/test_data_from_tournament.csv'
    no_of_games_in_tournament = 10
    no_of_random_players = 3
    generate_test_data_from_tournament(file_name, no_of_games_in_tournament, no_of_random_players)
    
    df = pd.read_csv(file_name)
    assert not df.empty, "DataFrame is empty after generating test data"
    b_vector = BoardVector()
    indices = b_vector.get_indices()
    for _, row in df.iterrows():
        # assert columns ranks do not cotain values larger than 4 or smaller than 1
        assert np.all(1 <= np.array(row.iloc[indices['ranks']])) and np.all(np.array(row.iloc[indices['ranks']]) <= 4), f"Row {row} contains ranks outside the range 1-4: {row.iloc[indices['ranks']]}"
        # assert columns turns do not contain values smaller than 0, but also are not all equal to 0
        assert np.all(np.array(row.iloc[indices['turns']]) >= 0), f"Row {row} contains turns smaller than 0: {row.iloc[indices['turns']]}"
    assert 0 in df['turns_before_end'].values, "Value 0 does not occur in 'turns_before_end' column"
    assert not np.all(df['turns_before_end'].values == 0), "All values in 'turns_before_end' column are zero"