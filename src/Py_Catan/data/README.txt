Training data generated on August 8.

Path to data: ../training_data/data_from_tournament_august_8.csv
Path to model: ../trained_models/model_trained_with_data_from_tournament_august_8.keras

Setting:
    N= 250
    for identifier in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'test']:
        print(f"Generating training data for identifier: {identifier}")
        path_to_data_for_this_round = path_to_data.replace('.csv', f'_{identifier}.csv')
        generate_test_data_from_tournament_with_random_actions(path_to_data_for_this_round,
                                            no_of_games_in_tournament=N,
                                            no_of_random_players=0,
                                            fraction_of_random_actions=0.35)
    print(f"Done generating training and test data.")

Training data generated on August 8.
    # === Train the model ===
    model_type.train_the_model(
        data=data, batch_size = 128, epochs = 256, verbose = 1)


This is copied to dirctorory src/Py_Catan/data 
- data as tournament_data_with_random_a.csv etc
- model as model_trained_on_limited_tournament_data.keras