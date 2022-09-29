import numpy as np
import tensorflow as tf


def compute_noise_metric(datapoint):

    roi = datapoint[:, : 200]
    return np.mean(roi)


def compute_mse(predictions, ground_test, noises=None):

    mse = tf.keras.metrics.MeanSquaredError()
    mse_results = []

    # For each datapoint to be tested, compute the MSE.
    for index, prediction in enumerate(predictions):
        mse.reset_state()
        mse.update_state(ground_test[index], prediction)

        result = (mse.result().numpy(), index)

        if noises is not None:
            result = (mse.result().numpy(), index, noises[index])

        mse_results.append(result)

    return mse_results


def load_batch_setup_and_eval(setup, run):

    # Load model
    model = tf.keras.models.load_model('saved_models/' + setup['name'] + '_batch' + str(run))
    model.summary()

    # Load data
    data_test = np.load('simulation\\datasets\\' + setup['dataset'] + '_dataset.npy')
    ground_test = np.load('simulation\\datasets\\' + setup['dataset'] + '_ground_truth.npy')
    sim_params = np.load('simulation\\datasets\\' + setup['dataset'] + '_params.npy')

    data_test = data_test[run]
    ground_test = ground_test[run]
    sim_params = sim_params[run]

    # Prep dataset for 2D CNN
    if len(model.layers[0].output_shape[1:]) == 3 and "imag" not in setup["name"]:
        data_test = np.expand_dims(data_test, 3)

    # Generate predictions
    predictions = model.predict(data_test)

    trainable_params = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    non_trainable_params = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params

    return data_test, ground_test, predictions, sim_params, total_params
