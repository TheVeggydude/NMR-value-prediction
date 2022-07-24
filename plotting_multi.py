import matplotlib.pyplot as plt
import numpy as np

import util

setups = [
    # {
    #     "name": "1d_cnn_v1_raw",
    #     "dataset": "simple_simulation_raw",
    #
    #     "batch_size": 32,
    #     "epochs": 10000,
    #     "runs": 10
    # },
    {
        "name": "1d_cnn_v1_proc",
        "dataset": "simple_simulation_proc",

        "batch_size": 32,
        "epochs": 10000,
        "runs": 10
    },
    # {
    #     "name": "1d_cnn_v4_k8",
    #     "dataset": "simple_simulation_proc",
    #
    #     "batch_size": 32,
    #     "epochs": 10000,
    #     "runs": 10
    # },
    {
        "name": "1d_cnn_v4",
        "dataset": "simple_simulation_proc",

        "batch_size": 32,
        "epochs": 10000,
        "runs": 10
    },
    # {
    #     "name": "1d_cnn_v4_k32",
    #     "dataset": "simple_simulation_proc",
    #
    #     "batch_size": 32,
    #     "epochs": 10000,
    #     "runs": 10
    # },
    # {
    #     "name": "1d_cnn_v4_k64",
    #     "dataset": "simple_simulation_proc",
    #
    #     "batch_size": 32,
    #     "epochs": 10000,
    #     "runs": 10
    # },
    {
        "name": "1d_cnn_v7",
        "dataset": "simple_simulation_proc",

        "batch_size": 32,
        "epochs": 10000,
        "runs": 10
    },
]

if __name__ == '__main__':
    all_predictions_for_setup = []

    for setup in setups:
        prediction_results = []

        for run in range(setup['runs']):
            data_test, ground_test, model, eval_score, predictions, _ = util.load_batch_setup_and_eval(setup, run)

            mse_results = util.compute_mse(predictions, ground_test)
            prediction_results.extend([x[0] for x in mse_results])

        prediction_results = np.asarray(prediction_results)

        # Scale MSE values to the same scale for comparison
        # prediction_results = prediction_results / (np.amax(prediction_results))

        all_predictions_for_setup.append(prediction_results)

    plt.hist(all_predictions_for_setup)

    plt.title("Prediction error distribution per setup")
    plt.legend([setup["name"] for setup in setups])
    plt.ylabel("Count")
    plt.xlabel("Mean Squared Error")
    plt.show()
