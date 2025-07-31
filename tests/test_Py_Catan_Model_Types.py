import sys  
sys.path.append("./src/Py_Catan")
sys.path.append("./src")
import pytest
import numpy as np
import pandas as pd
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
    assert not np.all(np.array(model_type.calculate_correlations()) > 0.8), "Correlations should not be high."
    model_type.read_test_data_from_file("./tests/test_models/training_data_for_specific_villages_and_streets_test.csv")
    assert np.all(np.array(model_type.calculate_correlations()) > 0.8), "Correlations should be high."

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