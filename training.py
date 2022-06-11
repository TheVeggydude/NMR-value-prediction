import numpy as np
import tensorflow as tf

import models

setups = [
    # {
    #     "name": "2d_cnn_simple_1filter",
    #     "dataset": "2022-02-19T16_52_00",
    #     "type": "cnn",
    #     "v": 8,
    #     "dimensions": ((301, 2), (301, 512, 1)),
    #     "n_rep": 1,
    #     "batch_size": 32,
    #     "epochs": 10000,
    #     "runs": 10
    # },
    # {
    #     "name": "2d_cnn_simple_8filters",
    #     "dataset": "2022-02-19T16_52_00",
    #     "type": "cnn",
    #     "v": 9,
    #     "dimensions": ((301, 2), (301, 512, 1)),
    #     "n_rep": 1,
    #     "batch_size": 32,
    #     "epochs": 10000,
    #     "runs": 10
    # },
    # {
    #     "name": "2d_cnn_simple_32filters",
    #     "dataset": "2022-02-19T16_52_00",
    #     "type": "cnn",
    #     "v": 11,
    #     "dimensions": ((301, 2), (301, 512, 1)),
    #     "n_rep": 1,
    #     "batch_size": 32,
    #     "epochs": 10000,
    #     "runs": 10
    # },
    # {
    #     "name": "1d_cnn_simple_kernel4",
    #     "dataset": "2022-02-19T16_52_00",
    #     "type": "cnn",
    #     "v": 3,
    #     "dimensions": ((301, 2), (301, 512)),
    #     "n_rep": 1,
    #     "batch_size": 32,
    #     "epochs": 10000,
    #     "runs": 10
    # },
    # {
    #     "name": "1d_cnn_simple_kernel8",
    #     "dataset": "2022-02-19T16_52_00",
    #     "type": "cnn",
    #     "v": 4,
    #     "dimensions": ((301, 2), (301, 512)),
    #     "n_rep": 1,
    #     "batch_size": 32,
    #     "epochs": 10000,
    #     "runs": 10
    # },
    # {
    #     "name": "1d_cnn_simple_kernel16",
    #     "dataset": "2022-02-19T16_52_00",
    #     "type": "cnn",
    #     "v": 5,
    #     "dimensions": ((301, 2), (301, 512)),
    #     "n_rep": 1,
    #     "batch_size": 32,
    #     "epochs": 10000,
    #     "runs": 10
    # },
    # {
    #     "name": "1d_cnn_simple_kernel32",
    #     "dataset": "2022-02-19T16_52_00",
    #     "type": "cnn",
    #     "v": 6,
    #     "dimensions": ((301, 2), (301, 512)),
    #     "n_rep": 1,
    #     "batch_size": 32,
    #     "epochs": 10000,
    #     "runs": 10
    # },
    # {
    #     "name": "1d_cnn_simple_kernel16",
    #     "dataset": "2022-02-19T16_52_00",
    #     "type": "cnn_api",
    #     "dimensions": ((301, 2), (301, 512)),
    #     "batch_size": 32,
    #     "epochs": 10000,
    #     "runs": 10
    # },
    {
        "name": "1d_cnn_batch_kernel16",
        "dataset": "2022-02-19T16_52_00",
        "type": "cnn_api_batch",
        "dimensions": ((301, 2), (301, 512)),
        "batch_size": 32,
        "epochs": 10000,
        "runs": 10
    },
]

K = 10

if __name__ == '__main__':

    # Train each setup in the list
    for setup in setups:

        print('Running setup:', setup['name'])

        ###
        # Data loading
        ###

        data = np.asarray([
            np.load(
                'simulation\\datasets\\' + setup['dataset'] + '_dataset_batch' + str(i) + '.npy'
            ) for i in range(K)
        ])

        ground = np.asarray([
            np.load(
                'simulation\\datasets\\' + setup['dataset'] + '_ground_truth_batch' + str(i) + '.npy'
            ) for i in range(K)
        ])

        # Prep dataset for 2D CNN
        if len(setup['dimensions'][1]) == 3:
            data = np.expand_dims(data, 4)

        ###
        # For each required batch we select as the testing data, we select a validation batch and join the remaining
        # data into a training set.
        ###

        for run in range(setup['runs']):

            print('Run:', str(run))

            # Wrap validation set back around to 0th batch if test set is at the end
            val_idx = run + 1 if run + 1 < len(data) - 1 else 0

            data_val = data[val_idx]
            ground_val = ground[val_idx]

            # Return copies of the original array without modifying said original
            data_train = np.delete(data, [run, val_idx], 0)
            ground_train = np.delete(ground, [run, val_idx], 0)

            # Reshape data to form one continuous array of data points
            data_train = np.reshape(data_train, (-1, ) + setup['dimensions'][1])
            ground_train = np.reshape(ground_train, (-1, ) + setup['dimensions'][0])

            ###
            # Create model
            ###

            model = None

            if setup["type"] == "mlp":
                model = models.mlp

            elif setup["type"] == "cnn":
                model = models.cnn

            elif setup["type"] == "hybrid":
                model = models.hybrid

            elif setup["type"] == "cnn_api":
                model = models.cnn_api

            elif setup["type"] == "cnn_api_batch":
                model = models.cnn_api_batch

            if setup["type"] in ["cnn_api", "cnn_api_batch"]:
                model = model.create_model(setup["dimensions"][1])

            else:
                model = model.create_model(
                    setup["dimensions"][0],
                    setup["dimensions"][1],
                    v=setup["v"],
                    n_rep=setup['n_rep']
                )

            model.compile(
                loss="mean_squared_error",
                optimizer='adam',
                metrics=[
                    tf.keras.metrics.RootMeanSquaredError(),
                    tf.keras.metrics.MeanAbsoluteError(),
                    tf.keras.metrics.MeanAbsolutePercentageError()
                ]
            )

            ###
            # Training
            ###

            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )

            history = model.fit(
                data_train,
                ground_train,
                batch_size=setup["batch_size"],
                epochs=setup["epochs"],
                validation_data=(data_val, ground_val),
                callbacks=[early_stopping]
            )

            ###
            # Store model
            ###
            model.save('saved_models/' + setup['name'] + '_batch' + str(run))
