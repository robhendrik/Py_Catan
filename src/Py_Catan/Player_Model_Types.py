from keras.models import Model
from keras.models import load_model, clone_model
from keras.layers import Input, Embedding, Reshape, Concatenate, Lambda, Dense
import matplotlib.pyplot as plt
from importlib.resources import files
from Py_Catan.BoardVector import BoardVector
import pandas as pd
import numpy as np
import sys  
import os
import warnings
sys.path.append("../src")

class PlayerModelTypes:
    """
    Class to handle different player model types.
    """
    # Path for data files belonging to source code
    DEFAULT_PATH = 'Py_Catan.data'

    # Standard test set used for correlation in all child classes
    DEFAULT_TEST_DATA_FILE = files(DEFAULT_PATH).joinpath('data_for_model_trained_on_limited_tournament_test.csv')

    # Default mode, usually overwritten in child classes
    PATH_TO_DEFAULT_MODEL = files(DEFAULT_PATH).joinpath('model_trained_on_limited_tournament_data.keras')

    # Data that can be used for training, including a test file generated in same way as the training data.
    DEFAULT_TRAINING_DATA = [files('Py_Catan.data').joinpath(f'tournament_data_with_random_{identifier}.csv') for identifier in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']]
    DEFAULT_TRAINING_DATA_TEST_SET = files(DEFAULT_PATH).joinpath('tournament_data_with_random_test.csv')
    def __init__(self, model_file_name: str = ''):
        '''
        Model has to be stored as Keras file in directory Py_Catan.data

        Args:
        - model_file_name (str): Name of the Keras model file to load. If empty, uses the default model.
        '''
        self.header = BoardVector().header()
        self.indices = BoardVector().indices

        # === Model ===
        if model_file_name == '':
            # === Try loading the default model ===
            self.path_to_keras_model = files(self.DEFAULT_PATH).joinpath(self.PATH_TO_DEFAULT_MODEL)   
            if self.path_to_keras_model.exists():
                self.load_model_from_file(self.path_to_keras_model)
                self.model.compile(optimizer='adam', loss='mean_squared_error')
            else:
                # If the default model file does not exist, create a new blank model
                # and warn the user.
                warnings.warn("Default model file not found in Py_Catan.data directory. Will create blank model", UserWarning)
                self.reset_model_to_new()
                self.path_to_keras_model = ''                
        else:
            # === Load the specified model file ===
            self.path_to_keras_model = files(self.DEFAULT_PATH).joinpath(model_file_name)
            if not self.path_to_keras_model.exists():
                raise FileNotFoundError(f"Model file {model_file_name} not found in Py_Catan.data directory. Use function "
                                        "'load_model_from_file()' to load a model from a different path.")
            self.load_model_from_file(self.path_to_keras_model)
            self.model.compile(optimizer='adam', loss='mean_squared_error')
                
        # === Test Data ===
        self.path_to_test_data = files(self.DEFAULT_PATH).joinpath(self.DEFAULT_TEST_DATA_FILE)
        if not self.path_to_test_data.exists():
            raise FileNotFoundError(f"Sample data file not found in Py_Catan.data directory.")
        self.data_for_test_and_correlation = pd.read_csv(self.path_to_test_data, header=0)

        # === Settings for ranking based on game statistics ===
        # This is used to calculate the y values for the model based on game statistics.
        # Ranking of players based on their performance
        self.score_table_for_ranking_per_game = [10, 5, 2, 0]
        self.discount_factor = 0.95  # Discount factor for future rewards
        pass

    def read_test_data_from_file(self, file_name: str):
        """
        Only used if you want to deviate from the default data defined in 
        self.path_to_sample_data
        
        Reads data from a CSV file and updates the internal 'data_for_test_and_correlation' attribute.
        """
        # Try to resolve the path either as a package resource or a direct file path
        try:
            path = files(file_name)
            if not path.exists():
                raise FileNotFoundError
        except Exception:
            path = file_name
            if not os.path.exists(path):
                raise FileNotFoundError(f"Data file {file_name} not found.")
        self.data_for_test_and_correlation = pd.read_csv(path, header=0)

    def reset_model_to_new(self) -> None:
        """
        Creates a new Keras model with the same structure as the current model.
        """
        vector = BoardVector()
        # === Define Model Inputs ===
        input1_layer = Input(shape=(len(vector.indices['nodes']),), dtype='int32', name='input1')
        input2_layer = Input(shape=(len(vector.indices['edges']),), dtype='int32', name='input2')
        input3_layer = Input(shape=(len(vector.indices['hands']),), dtype='float32', name='input3')

        # === Embedding Layers ===
        embed1 = Embedding(input_dim=9, output_dim=4, name='embed1')(input1_layer)  # shape: (None, 72, 4)
        embed2 = Embedding(input_dim=6, output_dim=3, name='embed2')(input2_layer)  # shape: (None, 54, 3)

        embed1_flat = Reshape((len(vector.indices['nodes']) * 4,), name='reshape1')(embed1)
        embed2_flat = Reshape((len(vector.indices['edges']) * 3,), name='reshape2')(embed2)

        # === Normalize input3 by dividing by 10 ===
        normalized_input3 = Lambda(lambda x: x / 10.0, name='normalize_input3')(input3_layer)

        # === Concatenate all inputs ===
        combined = Concatenate(name='concat')([embed1_flat, embed2_flat, normalized_input3])

        # === Fully Connected Layers ===
        x = Dense(128, activation='relu', name='dense1')(combined)
        x = Dense(64, activation='relu', name='dense2')(x)
        output = Dense(4, activation='linear', name='output')(x)

        # === Build and Compile Model ===
        model = Model(inputs=[input1_layer, input2_layer, input3_layer], outputs=output)
        model.compile(optimizer='adam', loss='mse')
        self.model = model
        return
    
    def generate_training_input_data_from_catan_vector_format(self, data: pd.DataFrame = None):
        """
        Generates training input data from a dataframe in Catan vector format.
        """
        vector = BoardVector()
        if data is None:
            data = self.data_for_test_and_correlation

        # === Extract inputs (from column 6 onward) ===
        # input1: first edges
        input1 = data.iloc[:, vector.indices['nodes']].values.astype(np.int32)
        # input2: next nodes
        input2 = data.iloc[:, vector.indices['edges']].values.astype(np.int32)
        # input3: final 4 hands
        input3 = data.iloc[:, vector.indices['hands']].values.astype(np.int32)
        return [input1, input2, input3]
    
    def train_the_model(self, 
                        data: pd.DataFrame = None,
                        batch_size: int = 128, 
                        epochs: int = 256,
                        verbose: int = 0) -> None:
        """
        Trains the model with the provided training data.

        Arguments:
        - data: DataFrame containing the training data as a list of BoardVectors. If None, uses self.data.
        - batch_size: Number of samples per gradient update.
        - epochs: Number of epochs to train the model.

        """
        if self.model is None:
            raise ValueError("Model is not initialized. Please load or create a model first.")
        if data is None:
            data = self.data_for_test_and_correlation

        x_train = self.generate_training_input_data_from_catan_vector_format(data)
        y_train = self.y_values_from_data(data)

        # Compile the model if not already compiled
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train the model
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose)
        return

    def load_model_from_file(self, path_to_keras_model: str) -> None:
        """
        Loads a Keras model from a specified file and updates the internal model attribute.
        path_keras_model can be a string or a files(string) object.
        """
        # Try to resolve the path either as a package resource or a direct file path
        try:
            path = files(path_to_keras_model)
            if not path.exists():
                raise FileNotFoundError
        except Exception:
            path = path_to_keras_model
            if not os.path.exists(path):
                raise FileNotFoundError(f"Data file {path_to_keras_model} not found.")
        new_model = load_model(path, safe_mode=False , compile=True)
        self.model = new_model
        return

    def save_model_to_file(self, path_to_keras_model: str) -> None:
        """
        Saves the current model to a specified file.
        """
        # Try to resolve the path either as a package resource or a direct file path
        # Check if the directory exists and is writable
        dir_path = os.path.dirname(path_to_keras_model)
        if dir_path and not os.path.exists(dir_path):
            raise FileNotFoundError(f"Directory {dir_path} does not exist for saving model.")
        if dir_path and not os.access(dir_path, os.W_OK):
            raise PermissionError(f"Directory {dir_path} is not writable for saving model.")
        path = path_to_keras_model
        self.model.save(path, save_format='tf', include_optimizer=True)
        return
    
    def get_model(self) -> Model:
        """
        Returns the current Keras model.
        """
        if self.model is None:
            raise ValueError("Model is not initialized. Please load or create a model first.")
        return self.model
    
    def y_values_from_data(self, data: pd.DataFrame = None):
        """
        Overwrite this in inheritance
        e.g., 
            return self._y_values_from_data_in_data_file() 
        or
            return self._y_values_from_properties()
        """
        return self._y_values_from_data_in_data_file(data)

    def _y_values_from_data_in_data_file(self, data: pd.DataFrame = None):
        """
        Returns the y value as stored in the sampled data from tournaments.
        (this represents the value for a value based player)

        This is what the jul 27 AI player models have been trained on.
        """
        if data is None:
            data = self.data_for_test_and_correlation
        value_indices = self.indices['values']
        return data.iloc[:, value_indices].values.astype(np.float32)
    
    def _y_values_from_tournament_statistic(self, data: pd.DataFrame = None):
        """
        Returns the y values generated from game statistics.
        """
        if data is None:
            data = self.data_for_test_and_correlation
        ranking_indices = self.indices['ranks']
        points = self._calculate_points_from_ranking(data.iloc[:, ranking_indices].values)
        turns_indices = self.indices['turns']
        discount = self._calculate_discount_factor(data.iloc[:, turns_indices].values)
        y = points * discount
        return y

    def _y_values_from_properties(self, data: pd.DataFrame = None):
        """
        Return the y value calculates as:
        - For first player number of streets
        - For second player number of villages
        - For third player number of towns      
        - For fourth player number of brick resource cards

        This is what the jul27 test models have been trained on.
        """
        if data is None:
            data = self.data_for_test_and_correlation
        y = []
        for _, row in data.iterrows():
            row_vector = np.array(row)
            edges = row_vector[self.indices['edges']]
            value_1 = np.count_nonzero(edges == 1)  # streets for player 1
            nodes = row_vector[self.indices['nodes']]
            value_2 = np.count_nonzero(nodes == 2) # villages for player 2
            value_3 = np.count_nonzero(nodes == 7) # towns for player 3
            hand = row_vector[self.indices['hand_for_player'][3]]
            value_4 = hand[0] # brick resource cards for player 4
            y.append([value_1, value_2, value_3, value_4])
        return np.array(y, dtype=np.float32)
    
    def _y_values_from_specific_villages_and_streets(self, data: pd.DataFrame = None):
        """
        Return the y value calculates as:
           - Player one trained to build village on node 0 and street on edge 0.
            - Player two trained to build village on node 10 and street on edge 16.
            - Player three trained to build village on node 20 and street on edge 25.
            Player four trained to build village on node 30 and street on edge 47.
        """
        if data is None:
            data = self.data_for_test_and_correlation
        y = []
        for _, row in data.iterrows():
            row_vector = np.array(row)
            edges = row_vector[self.indices['edges']]
            nodes = row_vector[self.indices['nodes']]
            value_1 = 1.0 * (1 if edges[0] == 1 else 0)  + 1.0 * (1 if nodes[0] == 1 else 0) # street and village for player 1
            value_2 = 1.0 * (1 if edges[16] == 2 else 0) + 1.0 * (1 if nodes[10] == 2 else 0) # street and village for player 2
            value_3 = 1.0 * (1 if edges[25] == 3 else 0) + 1.0 * (1 if nodes[20] == 3 else 0) # street and village for player 3
            value_4 = 1.0 * (1 if edges[47] == 4 else 0) + 1.0 * (1 if nodes[30] == 4 else 0) # street and village for player 4
            y.append([value_1, value_2, value_3, value_4])
        return np.array(y, dtype=np.float32)
    
    def y_values_from_value_function_based(self, data: pd.DataFrame = None):
        """
        Returns the y values calculated based on a value function.
        This is used for testing the model with a value function based player.
        """
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #
        #              TO
        #
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return

    
    def create_prediction_from_model(self):
        """
        Creates a prediction from the model using the data in 'self.data_for_test_and_correlation'.
        """
        # === Extract inputs ===
        # input1: first edges
        input1 = self.data_for_test_and_correlation.iloc[:, self.indices['nodes']].values.astype(np.int32)
        # input2: next nodes
        input2 = self.data_for_test_and_correlation.iloc[:, self.indices['edges']].values.astype(np.int32)
        # input3: final 4 hands
        input3 = self.data_for_test_and_correlation.iloc[:, self.indices['hands']].values.astype(np.int32)

        # === Sample Predictions ===
        return self.model.predict([input1, input2, input3], verbose=0)
    
    def create_prediction_from_vector(self, vector: BoardVector):
        """
        Creates a prediction from the model using a vector.
        """
        ivctr = vector.vector
        df = pd.DataFrame([ivctr], columns=vector.header())
        inputs = self.generate_training_input_data_from_catan_vector_format(df)
        return self.model.predict(inputs, verbose=0)

    def _create_plot_for_correlations(self, y_test: np.ndarray = None, y_pred: np.ndarray = None) -> list:
        """
        Plots the correlation between actual and predicted values.]
        Default y_pred is created from the 'self.model' using 'self.data'
        Default y_test is created from the 'self.data' using 'y_values_from_data()'.

        Function returns the 4 correlation coefficients for each output.
        """
        if y_test is None:
            y_test = self.y_values_from_data()
        if y_pred is None:
            y_pred = self.create_prediction_from_model()

        fig, axs = plt.subplots(2, 2, figsize=(10, 8))

        for i in range(4):
            ax = axs[i // 2, i % 2]
            ax.scatter(y_test[:, i], y_pred[:, i], alpha=0.5)
            ax.set_xlabel(f'Actual y_test[{i}]')
            ax.set_ylabel(f'Predicted y_pred[{i}]')
            ax.set_title(f'y_pred[{i}] vs y_test[{i}]')
            ax.grid(True)

        plt.tight_layout()
        plt.show()

        return self.calculate_correlations(y_test, y_pred)

    def calculate_correlations(self, y_test: np.ndarray = None, y_pred: np.ndarray = None) -> list:
        """
        Calculates the correlation coefficients between actual and predicted values.
        Default y_pred is created from the 'self.model' using 'self.data'
        Default y_test is created from the 'self.data' using 'y_values_from_data()'.
        
        Returns a list of correlation coefficients for each output.
        """
        if y_test is None:
            y_test = self.y_values_from_data()
        if y_pred is None:
            y_pred = self.create_prediction_from_model()

        correlations = []
        for i in range(4):
            corr = np.corrcoef(y_test[:, i], y_pred[:, i])[0, 1]
            correlations.append(corr)
        return correlations

    def _calculate_discount_factor(self, turns_before_end:  pd.DataFrame):
            """
            Calculates the discount factor based on the number of turns before the end of the game.
            """
            discount_factors = []
            for row in turns_before_end:
                if row <= 0:
                    factor = [1.0]
                else:
                    factor = self.discount_factor ** row
                discount_factors.append(factor)
            return np.array(discount_factors, dtype=np.float64)

    def _calculate_points_from_ranking(self,ranking: pd.DataFrame):
            '''
            Calculate the points for each player based on their results.
            The points are calculated based on the score table for ranking per game.    
            if multiple players have the same score, they will receive the average of the scores for those positions.
            '''
            score_table = self.score_table_for_ranking_per_game
            temp_table = np.sort(score_table.copy())
            temp_results = ranking.copy()
            points = []
            for row in temp_results:
                row_points = np.zeros(len(row), np.float64)
                temp_row = row.copy()
                temp_table_row = temp_table.copy()
                while max(temp_row) > 0:
                    max_value = max(temp_row)
                    indices = [i for i, j in enumerate(temp_row) if j == max_value]
                    score = sum(temp_table_row[:len(indices)]) / len(indices)
                    for i in indices:
                        row_points[i] = score
                        temp_row[i] = -1000
                    temp_table_row = temp_table_row[len(indices):]
                points.append(row_points)
            points = np.array(points, dtype=np.float64)
            return points

# ----------------------------------------------------------------------
class model_trained_on_streets_villages_towns_b_cards(PlayerModelTypes):
    def __init__(self):
        """
        Model used for testing, trained on specific behavior:
        - for first player number of streets
        - for second player number of villages
        - for third player number of towns
        - for fourth player number of b_cards

        The test data is sample test data from tournaments between value based players. If you want to change 
        use 'read_test_data_from_file()' to read test data from a file.
        """
        super().__init__(model_file_name="model_trained_on_streets_villages_towns_b_cards.keras")

        # === Do not load specific test data, use default ===
        # # === Load test data for this model ===
        # self.path_to_test_data = files(self.DEFAULT_PATH).joinpath('data_for_model_trained_on_streets_villages_towns_b_cards_test.csv')
        # if not self.path_to_test_data.exists():
        #     raise FileNotFoundError(f"Sample data file not found in Py_Catan.data directory.")
        # self.data_for_test_and_correlation = pd.read_csv(self.path_to_test_data, header=0)
        return
        return
    
    def plot_correlation(self):
        """
        Plots the correlation between actual and predicted values.
        """
        y_test = self.y_values_from_data()
        y_pred = self.create_prediction_from_model()
        return self._create_plot_for_correlations(y_test, y_pred)
    
    def y_values_from_data(self, data: pd.DataFrame = None):
        return self._y_values_from_properties(data)

# ----------------------------------------------------------------------
class model_trained_on_limited_tournament_data(PlayerModelTypes):

    def __init__(self):
        """
        Model on data generated on July 27 2025 from small tournament.

        The test data is sample test data from tournaments between value based players. If you want to change 
        use 'read_test_data_from_file()' to read test data from a file.
        """
        super().__init__(model_file_name="model_trained_on_limited_tournament_data.keras")

        # === Do not load specific test data, use default ===
        # # === Load test data for this model ===
        # # this data is randomly sampled from tournaments with random players and function based players
        # self.path_to_test_data = files(self.DEFAULT_PATH).joinpath('data_for_model_trained_on_limited_tournament_test.csv')
        # if not self.path_to_test_data.exists():
        #     raise FileNotFoundError(f"Sample data file not found in Py_Catan.data directory.")
        # self.data_for_test_and_correlation = pd.read_csv(self.path_to_test_data, header=0)
        return
    
    def plot_correlation(self):
        """
        Plots the correlation between actual and predicted values.
        """
        y_test = self.y_values_from_data()
        y_pred = self.create_prediction_from_model()
        return self._create_plot_for_correlations(y_test, y_pred) 
    
    def y_values_from_data(self, data: pd.DataFrame = None):
        return self._y_values_from_data_in_data_file(data)
    

# ----------------------------------------------------------------------
class blank_model(PlayerModelTypes):

    def __init__(self):
        """
        Untrained model cloned from the default model. Model has to be trained by the user:
        - Use 'model_type.train_the_model()' to train the model with data.

        After this you can plot correlations, save the model, make predictions, etc.

        The test data is an empty DataFrame, so no correlations can be plotted.
        """
        super().__init__()
        # this is a blank model, so everything has to be filled by user
        self.reset_model_to_new()
        self.path_to_test_data = ""
        self.path_to_keras_model = ""
        
        # === Do not load specific test data, use default ===
        self.data_for_test_and_correlation = pd.DataFrame()

        return
    
    def plot_correlation(self):
        """
        Plots the correlation between actual and predicted values.
        """
        y_test = self.y_values_from_data()
        y_pred = self.create_prediction_from_model()
        return self._create_plot_for_correlations(y_test, y_pred)
    
    def y_values_from_data(self, data: pd.DataFrame = None):
        return self._y_values_from_data_in_data_file(data)
    

# ----------------------------------------------------------------------    
class model_trained_on_specific_villages_and_streets(PlayerModelTypes):
    
    def __init__(self):
        """
        Model used for testing, trained on specific behavior:
        - Player one trained to build village on node 0 and street on edge 0.
        - Player two trained to build village on node 10 and street on edge 16.
        - Player three trained to build village on node 20 and street on edge 25.
        - Player four trained to build village on node 30 and street on edge 47.

        The test data is sample test data from tournaments between value based players. If you want to change 
        use 'read_test_data_from_file()' to read test data from a file.
        """
        super().__init__(model_file_name="model_trained_on_specific_villages_and_streets.keras")

        # === Do not load specific test data, use default ===
        # # === Load test data for this model ===
        # self.path_to_test_data = files('Py_Catan.data').joinpath("data_for_model_trained_on_specific_villages_and_streets_test.csv")
        # if not self.path_to_test_data.exists():
        #     raise FileNotFoundError(f"Test data file not found in Py_Catan.data directory.")
        # self.data_for_test_and_correlation = pd.read_csv(self.path_to_test_data, header=0)
        # return
    
    def plot_correlation(self):
        """
        Plots the correlation between actual and predicted values.
        """
        y_test = self._y_values_from_specific_villages_and_streets()
        y_pred = self.create_prediction_from_model()
        return self._create_plot_for_correlations(y_test, y_pred) 
    
    def y_values_from_data(self, data: pd.DataFrame = None):
        return self._y_values_from_specific_villages_and_streets(data)
