# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import numpy as np
import tensorflow as tf

import models

setups = [
    {
        "name": "imag_1block",
        "dataset": "imag",

        "model": models.cnn_2d.v2.create_model,
        "options": {
            'input_shape': (301, 512, 2),
            'output_shape': (301, 2, 1),
            'learning_rate': 0.001,
        },

        "batch_size": 8,
        "epochs": 10000,
        "runs": 10
    },
    {
        "name": "imag_2block",
        "dataset": "imag",

        "model": models.cnn_2d.v10.create_model,
        "options": {
            'input_shape': (301, 512, 2),
            'output_shape': (301, 2, 1),
            'learning_rate': 0.001,
            'filters': [
                128,
                1,
            ],
            'kernels': [
                (10, 24),
                (10, 2),
            ],
            'block_length': 1,
        },

        "batch_size": 8,
        "epochs": 10000,
        "runs": 10
    },
    {
        "name": "imag_2block_4cnn",
        "dataset": "imag",

        "model": models.cnn_2d.v10.create_model,
        "options": {
            'input_shape': (301, 512, 2),
            'output_shape': (301, 2, 1),
            'learning_rate': 0.001,
            'filters': [
                16,
                1,
            ],
            'kernels': [
                (10, 24),
                (10, 2),
            ],
            'block_length': 2,
        },

        "batch_size": 8,
        "epochs": 10000,
        "runs": 10
    },
    {
        "name": "imag_3block",
        "dataset": "imag",

        "model": models.cnn_2d.v3.create_model,
        "options": {
            'input_shape': (301, 512, 2),
            'output_shape': (301, 2, 1),
            'learning_rate': 0.001,
            'filters': [
                32,
                16,
            ],
            'kernels': [
                (10, 24),
                (10, 10),
                (10, 2),
            ],
            'block_length': 1,
        },

        "batch_size": 8,
        "epochs": 10000,
        "runs": 10
    },
    {
        "name": "imag_4block",
        "dataset": "imag",

        "model": models.cnn_2d.v8.create_model,
        "options": {
            'input_shape': (301, 512, 2),
            'output_shape': (301, 2, 1),
            'learning_rate': 0.001,
            'filters': [
                32,
                16,
                8
            ],
            'kernels': [
                (10, 24),
                (10, 10),
                (10, 2),
                (10, 2),
            ],
            'block_length': 1,
        },

        "batch_size": 8,
        "epochs": 10000,
        "runs": 10
    },
    # {
    #     "name": "imag_h_0_5_w_0_5",
    #     "dataset": "imag",
    #
    #     "model": models.cnn_2d.v11.create_model,
    #     "options": {
    #         'input_shape': (301, 512, 2),
    #         'output_shape': (301, 2, 1),
    #         'learning_rate': 0.001,
    #         'filters': [
    #             32,
    #             16,
    #         ],
    #         'kernels': [
    #             (5, 12),
    #             (5, 5),
    #             (5, 2),
    #         ],
    #     },
    #
    #     "batch_size": 8,
    #     "epochs": 10000,
    #     "runs": 10
    # },
    # {
    #     "name": "imag_h_0_5_w_1_0",
    #     "dataset": "imag",
    #
    #     "model": models.cnn_2d.v11.create_model,
    #     "options": {
    #         'input_shape': (301, 512, 2),
    #         'output_shape': (301, 2, 1),
    #         'learning_rate': 0.001,
    #         'filters': [
    #             32,
    #             16,
    #         ],
    #         'kernels': [
    #             (5, 24),
    #             (5, 10),
    #             (5, 2),
    #         ],
    #     },
    #
    #     "batch_size": 8,
    #     "epochs": 10000,
    #     "runs": 10
    # },
    # {
    #     "name": "imag_h_0_5_w_2_0",
    #     "dataset": "imag",
    #
    #     "model": models.cnn_2d.v11.create_model,
    #     "options": {
    #         'input_shape': (301, 512, 2),
    #         'output_shape': (301, 2, 1),
    #         'learning_rate': 0.001,
    #         'filters': [
    #             32,
    #             16,
    #         ],
    #         'kernels': [
    #             (5, 48),
    #             (5, 20),
    #             (5, 2),
    #         ],
    #     },
    #
    #     "batch_size": 8,
    #     "epochs": 10000,
    #     "runs": 10
    # },
    # {
    #     "name": "imag_h_0_5_w_3_0",
    #     "dataset": "imag",
    #
    #     "model": models.cnn_2d.v11.create_model,
    #     "options": {
    #         'input_shape': (301, 512, 2),
    #         'output_shape': (301, 2, 1),
    #         'learning_rate': 0.001,
    #         'filters': [
    #             32,
    #             16,
    #         ],
    #         'kernels': [
    #             (5, 72),
    #             (5, 30),
    #             (5, 2),
    #         ],
    #     },
    #
    #     "batch_size": 8,
    #     "epochs": 10000,
    #     "runs": 10
    # },
    # {
    #     "name": "imag_h_0_5_w_3_0",
    #     "dataset": "imag",
    #
    #     "model": models.cnn_2d.v11.create_model,
    #     "options": {
    #         'input_shape': (301, 512, 2),
    #         'output_shape': (301, 2, 1),
    #         'learning_rate': 0.001,
    #         'filters': [
    #             32,
    #             16,
    #         ],
    #         'kernels': [
    #             (5, 96),
    #             (5, 40),
    #             (5, 2),
    #         ],
    #     },
    #
    #     "batch_size": 8,
    #     "epochs": 10000,
    #     "runs": 10
    # },
    # {
    #     "name": "imag_h_2_0_w_3_0",
    #     "dataset": "imag",
    #
    #     "model": models.cnn_2d.v11.create_model,
    #     "options": {
    #         'input_shape': (301, 512, 2),
    #         'output_shape': (301, 2, 1),
    #         'learning_rate': 0.001,
    #         'filters': [
    #             32,
    #             16,
    #         ],
    #         'kernels': [
    #             (20, 72),
    #             (20, 30),
    #             (20, 2),
    #         ],
    #     },
    #
    #     "batch_size": 8,
    #     "epochs": 10000,
    #     "runs": 10
    # },
    # {
    #     "name": "imag_h_1_0_w_4_0",
    #     "dataset": "imag",
    #
    #     "model": models.cnn_2d.v11.create_model,
    #     "options": {
    #         'input_shape': (301, 512, 2),
    #         'output_shape': (301, 2, 1),
    #         'learning_rate': 0.001,
    #         'filters': [
    #             32,
    #             16,
    #         ],
    #         'kernels': [
    #             (10, 96),
    #             (10, 40),
    #             (10, 2),
    #         ],
    #     },
    #
    #     "batch_size": 8,
    #     "epochs": 10000,
    #     "runs": 10
    # },
    # {
    #     "name": "imag_h_1_0_w_3_0",
    #     "dataset": "imag",
    #
    #     "model": models.cnn_2d.v11.create_model,
    #     "options": {
    #         'input_shape': (301, 512, 2),
    #         'output_shape': (301, 2, 1),
    #         'learning_rate': 0.001,
    #         'filters': [
    #             32,
    #             16,
    #         ],
    #         'kernels': [
    #             (10, 72),
    #             (10, 30),
    #             (10, 2),
    #         ],
    #     },
    #
    #     "batch_size": 8,
    #     "epochs": 10000,
    #     "runs": 10
    # },
    # {
    #     "name": "imag_h_1_0_w_2_0",
    #     "dataset": "imag",
    #
    #     "model": models.cnn_2d.v11.create_model,
    #     "options": {
    #         'input_shape': (301, 512, 2),
    #         'output_shape': (301, 2, 1),
    #         'learning_rate': 0.001,
    #         'filters': [
    #             32,
    #             16,
    #         ],
    #         'kernels': [
    #             (10, 48),
    #             (10, 20),
    #             (10, 2),
    #         ],
    #     },
    #
    #     "batch_size": 8,
    #     "epochs": 10000,
    #     "runs": 10
    # },
    # {
    #     "name": "imag_h_0_5_w_3_0",
    #     "dataset": "imag",
    #
    #     "model": models.cnn_2d.v11.create_model,
    #     "options": {
    #         'input_shape': (301, 512, 2),
    #         'output_shape': (301, 2, 1),
    #         'learning_rate': 0.001,
    #         'filters': [
    #             32,
    #             16,
    #         ],
    #         'kernels': [
    #             (5, 72),
    #             (5, 30),
    #             (5, 2),
    #         ],
    #     },
    #
    #     "batch_size": 8,
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

        data = np.load('simulation/datasets/' + setup['dataset'] + '_dataset.npy')
        ground = np.load('simulation/datasets/' + setup['dataset'] + '_ground_truth.npy')

        ###
        # For each required batch we select as the testing data, we select a validation batch and join the remaining
        # data into a training set.
        ###

        for run in range(setup['runs']):

            print(setup['name'] + ' - run: ' + str(run))

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

            optimizer = tf.keras.optimizers.Adam(
                learning_rate=setup["options"]["learning_rate"]
            )

            model.compile(
                loss="mean_squared_error",
                optimizer=optimizer,
                metrics=[
                    tf.keras.metrics.RootMeanSquaredError(),
                    tf.keras.metrics.MeanAbsoluteError(),
                    tf.keras.metrics.MeanAbsolutePercentageError()
                ],
                jit_compile=True
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
