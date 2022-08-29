# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import numpy as np
import tensorflow as tf

import models

setups = [
    {
        "name": "2d_cnn_v1",
        "dataset": "simple_sim_norm",

        "model": models.cnn_2d.v1.create_model,
        "options": {
            'input_shape': (301, 512, 1),
            'output_shape': (301, 2, 1),
            'learning_rate': 0.001,
            'filters': [
                32,
                16
            ],
            'kernels': [
                (10, 24),
                (10, 10),
                (10, 2),
            ]
        },

        "batch_size": 8,
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
                # jit_compile=True
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
