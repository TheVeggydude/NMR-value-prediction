import numpy as np
import tensorflow as tf


def compute_noise_metric(datapoint):

    roi = datapoint[:, : 200]
    return np.var(roi)


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
    data_test = np.load(
        'simulation\\datasets\\' + setup['dataset'] + '_dataset_batch' + str(run) + '.npy'
    )

    ground_test = np.load(
        'simulation\\datasets\\' + setup['dataset'] + '_ground_truth_batch' + str(run) + '.npy'
    )

    # Prep dataset for 2D CNN
    if len(model.layers[0].output_shape[1:]) == 3:
        data_test = np.expand_dims(data_test, 3)

    # Evaluate model
    eval_score = model.evaluate(data_test, ground_test, batch_size=setup["batch_size"])

    # Generate predictions
    predictions = model.predict(data_test)

    noises = [compute_noise_metric(dp) for dp in data_test]

    return data_test, ground_test, model, eval_score, predictions, noises
