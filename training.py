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
    # {
    #     "name": "1d_cnn_batch_kernel16",
    #     "dataset": "2022-02-19T16_52_00",
    #     "type": "cnn_api_batch",
    #     "dimensions": ((301, 2), (301, 512)),
    #     "batch_size": 32,
    #     "epochs": 10000,
    #     "runs": 10
    # },
    # {
    #     "name": "1d_cnn_v1_raw",
    #     "dataset": "simple_simulation_raw",
    #
    #     "model": models.cnn_1d.v1.create_model,
    #     "options": {
    #         "input_shape": (301, 512),
    #         "kernel_size": 16
    #     },
    #
    #     "batch_size": 32,
    #     "epochs": 10000,
    #     "runs": 10
    # },
    # {
    #     "name": "1d_cnn_v1_proc",
    #     "dataset": "simple_simulation_proc",
    #
    #     "model": models.cnn_1d.v1.create_model,
    #     "options": {
    #         "input_shape": (301, 512),
    #         "kernel_size": 16
    #     },
    #
    #     "batch_size": 32,
    #     "epochs": 10000,
    #     "runs": 10
    # },
    # {
    #     "name": "1d_cnn_v2",
    #     "dataset": "simple_simulation_proc",
    #
    #     "model": models.cnn_1d.v2.create_model,
    #     "options": {
    #         "input_shape": (301, 512),
    #         "kernel_size": 16
    #     },
    #
    #     "batch_size": 32,
    #     "epochs": 10000,
    #     "runs": 10
    # },
    # {
    #     "name": "1d_cnn_v3",
    #     "dataset": "simple_simulation_proc",
    #
    #     "model": models.cnn_1d.v3.create_model,
    #     "options": {
    #         "input_shape": (301, 512),
    #         "kernel_size": 16
    #     },
    #
    #     "batch_size": 32,
    #     "epochs": 10000,
    #     "runs": 10
    # },
    # {
    #     "name": "1d_cnn_v4",
    #     "dataset": "simple_simulation_proc",
    #
    #     "model": models.cnn_1d.v4.create_model,
    #     "options": {
    #         "input_shape": (301, 512),
    #         "kernel_size": 16
    #     },
    #
    #     "batch_size": 32,
    #     "epochs": 10000,
    #     "runs": 10
    # },
    # {
    #     "name": "1d_cnn_v7",
    #     "dataset": "simple_simulation_proc",
    #
    #     "model": models.cnn_1d.v7.create_model,
    #     "options": {
    #         "input_shape": (301, 512),
    #         "kernel_size": 16
    #     },
    #
    #     "batch_size": 32,
    #     "epochs": 10000,
    #     "runs": 10
    # },
    # {
    #     "name": "1d_cnn_v4_k32",
    #     "dataset": "simple_simulation_proc",
    #
    #     "model": models.cnn_1d.v4.create_model,
    #     "options": {
    #         "input_shape": (301, 512),
    #         "kernel_size": 32
    #     },
    #
    #     "batch_size": 32,
    #     "epochs": 10000,
    #     "runs": 10
    # },
    # {
    #     "name": "1d_cnn_v4_k64",
    #     "dataset": "simple_simulation_proc",
    #
    #     "model": models.cnn_1d.v4.create_model,
    #     "options": {
    #         "input_shape": (301, 512),
    #         "kernel_size": 64
    #     },
    #
    #     "batch_size": 32,
    #     "epochs": 10000,
    #     "runs": 10
    # },
    # {
    #     "name": "1d_cnn_v5",
    #     "dataset": "simple_simulation_proc",
    #
    #     "model": models.cnn_1d.v5.create_model,
    #     "options": {
    #         "input_shape": (301, 512),
    #         "kernel_size": 16
    #     },
    #
    #     "batch_size": 32,
    #     "epochs": 10000,
    #     "runs": 10
    # },
    # {
    #     "name": "1d_cnn_v6",
    #     "dataset": "simple_simulation_proc",
    #
    #     "model": models.cnn_1d.v6.create_model,
    #     "options": {
    #         "input_shape": (301, 512),
    #         "kernel_size": 32
    #     },
    #
    #     "batch_size": 32,
    #     "epochs": 10000,
    #     "runs": 10
    # },
    {
        "name": "2d_cnn_v3",
        "dataset": "simple_simulation_proc",

        "model": models.cnn_2d.v3.create_model,
        "options": {
            # 'filters': 1,
            'input_shape': (301, 512, 1),
            'output_shape': (301, 2, 1),
        },

        "batch_size": 32,
        "epochs": 10000,
        "runs": 10
    },
    # {
    #     "name": "2d_cnn_v4_cropped",
    #     "dataset": "simple_simulation_proc_cropped",
    #
    #     "model": models.cnn_2d.v4.create_model,
    #     "options": {
    #         'filters': 2,
    #         'input_shape': (301, 55, 1),
    #         'output_shape': (301, 2, 1),
    #     },
    #
    #     "batch_size": 32,
    #     "epochs": 10000,
    #     "runs": 10
    # },
]

K = 10

if __name__ == '__main__':

    # Train each setup in the list
    for setup in setups:

        print('Running setup:', setup['name'])

        ###
        # Data loading
        ###

        data = np.load('simulation\\datasets\\' + setup['dataset'] + '_dataset.npy')
        ground = np.load('simulation\\datasets\\' + setup['dataset'] + '_ground_truth.npy')

        # Prep dataset for 2D CNN
        if len(setup['options']['input_shape']) == 3:
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
            data_train = np.reshape(data_train, (-1, ) + setup['options']['input_shape'])
            ground_train = np.reshape(ground_train, (-1, ) + setup['options']['output_shape'])

            ###
            # Create model
            ###

            model = setup["model"](setup['options'])

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
