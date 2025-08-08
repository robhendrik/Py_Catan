import sys, os, time
sys.path.append("./src/Py_Catan")
sys.path.append("./src")
import pytest
import numpy as np
import pandas as pd
from importlib.resources import files
from Py_Catan import *

from Py_Catan.BoardStructure import BoardStructure, BoardLayout
from Py_Catan.PlotBoard import PlotCatanBoard
from Py_Catan.Preferences import PlayerPreferences
from Py_Catan.Player import Player
from Py_Catan.Board import Board
from Py_Catan.PlayerValueFunctionBased import Player_Value_Function_Based
from Py_Catan.PlayerModelBased import Player_Model_Based
from keras.models import Model
from keras.saving import load_model

import matplotlib.pyplot as plt

def test_PlayerModelTypes():
    """
    Test function for PlayerModelTypes class.
    It checks if the class can be instantiated and if the model types are correctly set.
    It checks if the class can be instantiated and if the model types are correctly set.
    """
    model_type = PlayerModelTypes()
    df = pd.DataFrame([[1,1,2,2],
                    [1,2,3,4],
                    [1,1,1,1],
                    [4,3,2,1]]
    )
    expected =np.array([[ 7.5,   7.5,   1.,    1.,  ],
                        [10.,    5.,    2.,    0.,  ],
                        [ 4.25,  4.25,  4.25,  4.25,],
                        [ 0.,    2.,5.,   10.,  ]])

    res = model_type._calculate_points_from_ranking(df.iloc[:, [0,1,2,3]].values)
    assert np.allclose(res, expected)

def test_calculate_discount_factor():
    """
    Test function for _calculate_discount_factor method.
    It checks if the discount factors are calculated correctly.
    """
    model_type = PlayerModelTypes()
    df = pd.DataFrame([[1],
                    [1],
                    [0],
                    [2]]
    )
    expected =np.array([[ 0.95,],
                        [ 0.95,],
                        [ 1.,],
                        [ 0.95*0.95,]])

    res = model_type._calculate_discount_factor(df.iloc[:, [0]].values)
    assert np.allclose(res, expected)

def test_values_from_tournament_statistic():
    """
    Test function for _y_values_from_tournament_statistic method.
    It checks if the values are calculated correctly from tournament statistics.
    """
    vctr = BoardVector()
    header = vctr.header()
    indices = vctr.get_indices()
    row_vector = np.zeros(len(header))
    df = pd.DataFrame([row_vector], columns=header)
    row_vector = np.zeros(len(header))
    row_vector[indices['turns']] = 1
    ranking = np.array([1, 2, 3, 4])
    row_vector[indices['ranks']] = ranking
    for _ in range(4):
        df.loc[len(df)] = row_vector
    model_type = PlayerModelTypes()

    points = model_type._calculate_points_from_ranking(df.iloc[:, indices['ranks']].values)
    discount = model_type._calculate_discount_factor(df.iloc[:, indices['turns']].values)
    expected = np.array([[0.,   0.,   0.,   0.,  ],
    [9.5,  4.75, 1.9,  0.,  ],
    [9.5,  4.75, 1.9,  0.,  ],
    [9.5,  4.75, 1.9,  0.,  ],
    [9.5,  4.75, 1.9,  0.,  ]])
    assert np.allclose(model_type._y_values_from_tournament_statistic(df), expected)

def test_y_values_from_specific_villages_and_streets():
    """
    Test function for _y_values_from_specific_villages_and_streets method.
    It checks if the values are calculated correctly from specific villages and streets.
    """
    vctr = BoardVector()
    header = vctr.header()
    indices = vctr.get_indices()
    row_vector = np.zeros(len(header))
    df = pd.DataFrame([row_vector], columns=header)
    row_vector = np.zeros(len(header))
    row_vector[indices['edge_0']] = 1
    row_vector[indices['node_0']] = 1
    df.loc[len(df)] = row_vector
    row_vector = np.zeros(len(header))
    row_vector[indices['edge_16']] = 2
    row_vector[indices['node_10']] = 2
    df.loc[len(df)] = row_vector
    row_vector = np.zeros(len(header))
    row_vector[indices['edge_25']] = 3
    row_vector[indices['node_20']] = 3
    row_vector[indices['edge_47']] = 4
    row_vector[indices['node_30']] = 4
    df.loc[len(df)] = row_vector
    row_vector = np.zeros(len(header))
    row_vector[indices['node_0']] = 1
    row_vector[indices['node_10']] = 2
    row_vector[indices['node_20']] = 3
    row_vector[indices['node_30']] = 4
    df.loc[len(df)] = row_vector
    model_type = PlayerModelTypes()

    expected = np.array([[ 0.,  0.,  0.,  0.,],
                        [ 2.,  0.,  0.,  0.,], 
                        [ 0.,  2.,  0.,  0.,],
                        [ 0.,  0.,  2.,  2.,],
                        [ 1.,  1.,  1.,  1.,]])
        
    assert np.allclose(model_type._y_values_from_specific_villages_and_streets(df), expected)

def test_training_model():
    """
    Test function for training the model.
    """
    model_type = blank_model()

    # === Load data ===
    file_name = "data_for_model_training_jul_27.csv"
    path_for_data = "./tests/test_models/"
    csv_files = [path_for_data + 'a_' + file_name]
    data = pd.concat([pd.read_csv(f, header=0) for f in csv_files], ignore_index=True)
    # Save initial weights right after model creation
    initial_weights = model_type.model.get_weights()
    model_type.train_the_model(data, epochs=1, batch_size=32)
    # Save initial weights right after model creation
    trained_weights = model_type.model.get_weights()

    # Check if weights have changed
    weights_changed = any(
        (init_w != trained_w).any() 
        for init_w, trained_w in zip(initial_weights, trained_weights)
    )

    assert weights_changed, "Model weights have not changed — model may not be trained"
    assert isinstance(model_type.model, Model), "Model should be an instance of tf.keras.Model"
    model_type.reset_model_to_new()
    # Save initial weights right after model creation
    reset_weights = model_type.model.get_weights()
    # Check if weights have changed
    weights_changed = any(
        (init_w != trained_w).any() 
        for init_w, trained_w in zip(reset_weights, trained_weights)
    )
    assert weights_changed, "Model weights have not changed — model may not be reset"

def test_calculate_correlations():
    """
    Test function for calculate_correlations method.
    It checks if the correlations are calculated correctly.
    """
    model_type = model_trained_on_streets_villages_towns_b_cards()
    corrs = model_type.calculate_correlations()
    assert np.all(np.array(corrs) > 0.8), "Correlations is too low"  # Check if all correlations are above 0.8

    model_type = model_trained_on_limited_tournament_data()
    corrs = model_type.calculate_correlations()
    assert np.all(np.array(corrs) > 0.8), "Correlations is too low"  # Check if all correlations are above 0.8

    model_type = model_trained_on_streets_villages_towns_b_cards()
    model_type.model = model_trained_on_limited_tournament_data().get_model()
    corrs = model_type.calculate_correlations()
    assert not np.all(np.array(corrs) > 0.8), "Correlations is too high"  # Check if all correlations are above 0.8

    model_type = model_trained_on_specific_villages_and_streets()
    assert np.all(np.array(model_type.calculate_correlations()) > 0.8), "Correlations should not be high."
   

def test_save_load_model():
    """
    Test function for saving and loading the model.
    """
    model_type = model_trained_on_limited_tournament_data()
    corrs = model_type.calculate_correlations()
    assert np.all(np.array(corrs) > 0.8), "Correlations is too low"  # Check if all correlations are above 0.8
    model_type.save_model_to_file("./tests/test_models/test_name_for_saving_and_loading.keras")

    model_type_2 = model_trained_on_streets_villages_towns_b_cards()
    corrs = model_type_2.calculate_correlations()
    assert np.all(np.array(corrs) > 0.8), "Correlations is too low"  # Check if all correlations are above 0.8

    model_type_2.load_model_from_file("./tests/test_models/test_name_for_saving_and_loading.keras")
    corrs = model_type_2.calculate_correlations()
    assert not np.all(np.array(corrs) > 0.8), "Correlations is too high"  # Check if all correlations are above 0.8

    model_type = model_trained_on_streets_villages_towns_b_cards()
    file_name = "test_name_for_saving_and_loading.keras"
    path_for_data = "./tests/test_models/"
    model_type.save_model_to_file(path_for_data + file_name)
    
    # Load the model
    loaded_model_type = PlayerModelTypes()
    loaded_model_type.load_model_from_file(path_for_data + file_name)
    def compare_keras_models(m1, m2, atol=1e-6, rtol=1e-6):
        if m1.to_json() != m2.to_json():
            return False, "Model architectures differ"
        
        w1 = m1.get_weights()
        w2 = m2.get_weights()
        if len(w1) != len(w2):
            return False, "Number of weight arrays differ"
        
        for i, (a, b) in enumerate(zip(w1, w2)):
            if not np.allclose(a, b, atol=atol, rtol=rtol):
                return False, f"Weights at index {i} differ"
        return True, "Models are equal within tolerance"
    
    # Check if the loaded model is the same as the original
    are_equal, message = compare_keras_models(model_type.model, loaded_model_type.model)
    assert are_equal, message

def test_different_model_types():
    """
    Test function for different model types.
    It checks if the model types are correctly set and if the correlations are calculated correctly.
    """
    model_type = model_trained_on_limited_tournament_data()
    assert model_type.path_to_test_data == files('Py_Catan.data').joinpath('data_for_model_trained_on_limited_tournament_test.csv')
    assert model_type.path_to_keras_model == files('Py_Catan.data').joinpath('model_trained_on_limited_tournament_data.keras')
    assert np.all(np.array(model_type.calculate_correlations()) > 0.9)

    model_type = model_trained_on_specific_villages_and_streets()
    assert model_type.path_to_test_data == files('Py_Catan.data').joinpath('data_for_model_trained_on_limited_tournament_test.csv')
    assert model_type.path_to_keras_model == files('Py_Catan.data').joinpath('model_trained_on_specific_villages_and_streets.keras')
    assert np.all(np.array(model_type.calculate_correlations()) > 0.9)

    model_type = model_trained_on_streets_villages_towns_b_cards()
    assert model_type.path_to_test_data == files('Py_Catan.data').joinpath('data_for_model_trained_on_limited_tournament_test.csv')
    assert model_type.path_to_keras_model == files('Py_Catan.data').joinpath('model_trained_on_streets_villages_towns_b_cards.keras')
    assert np.all(np.array(model_type.calculate_correlations()) > 0.9)

    model_type = blank_model()
    assert model_type.path_to_test_data == ''
    assert model_type.path_to_keras_model == ''
    # assert that calculating correlations raises an exception since no test data is loaded
    with pytest.raises(Exception):
        model_type.calculate_correlations()

def test_model_types_with_separate_test_data():
    # Test the model types and their properties with separate test data and default model files.
    model_type = model_trained_on_limited_tournament_data()
    model_type.read_test_data_from_file('./src/Py_Catan/data/data_for_model_trained_on_limited_tournament_test.csv')
    assert np.all(np.array(model_type.calculate_correlations()) > 0.8)
    model_type = model_trained_on_specific_villages_and_streets()
    model_type.read_test_data_from_file('./src/Py_Catan/data/data_for_model_trained_on_specific_villages_and_streets_test.csv')
    assert np.all(np.array(model_type.calculate_correlations()) > 0.8)
    model_type = model_trained_on_streets_villages_towns_b_cards()
    model_type.read_test_data_from_file('./src/Py_Catan/data/data_for_model_trained_on_streets_villages_towns_b_cards_test.csv')
    assert np.all(np.array(model_type.calculate_correlations()) > 0.8)

def test_plot_correlation_calls(monkeypatch):
    """
    Test that plot_correlation calls the underlying plot function without error,
    but does not actually display or save the plot.
    """


    # Patch plt.show and plt.savefig to prevent actual plotting

    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    monkeypatch.setattr(plt, "savefig", lambda *args, **kwargs: None)

    # Plot the correlations with the default test data.
    model_type = model_trained_on_limited_tournament_data()
    model_type.plot_correlation();
    model_type = model_trained_on_specific_villages_and_streets()
    model_type.plot_correlation();
    model_type = model_trained_on_streets_villages_towns_b_cards()
    model_type.plot_correlation();

    # assert that calculating correlations raises an exception since no test data is loaded
    with pytest.raises(Exception):
        model_type_3 = blank_model()
        model_type_3.plot_correlation();

def test_generate_test_data_and_train_model_from_tournament_with_random_actions(monkeypatch):

    # Patch plt.show and plt.savefig to prevent actual plotting
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    monkeypatch.setattr(plt, "savefig", lambda *args, **kwargs: None)

    # === Check if the directory for the data file exists and is writable ===
    path_to_data = './training_data/demo_data_from_tournament.csv'
    directory = os.path.dirname(path_to_data)
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    if not os.access(directory, os.W_OK):
        raise PermissionError(f"Directory is not writable: {directory}")

    # === Generate test data for the blank model demo training ===
    # add some randomness to also create data for options not chosen by the players
    N= 1
    for identifier in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'test']:
        print(f"Generating training data for identifier: {identifier}")
        path_to_data_for_this_round = path_to_data.replace('.csv', f'_{identifier}.csv')
        generate_test_data_from_tournament_with_random_actions(path_to_data_for_this_round,
                                            no_of_games_in_tournament=N,
                                            no_of_random_players=0,
                                            fraction_of_random_actions=0.35)
    print(f"Done generating training and test data.")

    # === Create a blank model instance ===
    model_type = blank_model()

    # === Define how you want to generate y-values (target for training) from the data ===
    # (some build in methods, but you can overwrite this with your own function
    # This function will be used for training, and for calculating correlations)
    model_type.y_values_from_data = model_type._y_values_from_data_in_data_file

    # === Read training data from the files ===
    files = [path_to_data.replace('.csv', f'_{identifier}.csv') for identifier in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']]
    data = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    assert data.shape[1] == 159

    # === Train the model ===
    model_type.train_the_model(
        data=data, batch_size = 128, epochs = 100, verbose = 1)

    # === Plot correlations for the trained model with default test_data ===
    model_type.read_test_data_from_file(model_type.DEFAULT_TEST_DATA_FILE)
    model_type.plot_correlation()
    assert np.all(np.array(model_type.calculate_correlations()) > 0.5)

    # === Plot correlations for the trained model with test_data generated together with the training data ===
    model_type.read_test_data_from_file(path_to_data.replace('.csv', f'_test.csv'))
    model_type.plot_correlation()
    assert np.all(np.array(model_type.calculate_correlations()) > 0.5)

    # === Save the trained model to a file ===
    path_to_model = './trained_models/model_trained_with_demo_data_from_tournament.keras'
    directory = os.path.dirname(path_to_model)
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    if not os.access(directory, os.W_OK):
        raise PermissionError(f"Directory is not writable: {directory}")

    if os.path.exists(path_to_model): # remove file if it exists, to check if saving works
        os.remove(path_to_model)
        assert not os.path.exists(path_to_model), f"Model file still exists after removal: {path_to_model}"
    model_type.save_model_to_file(path_to_model)

    # Assert the model file exists and is not empty
    assert os.path.exists(path_to_model), f"Model file not found: {path_to_model}"
    assert os.path.getsize(path_to_model) > 0, f"Model file is empty: {path_to_model}"