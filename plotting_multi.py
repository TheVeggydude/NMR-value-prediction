import matplotlib.pyplot as plt
import numpy as np

import scipy.stats
import util

setups = [
    {
        "name": "2d_cnn_v1",
        "display": "1 layer",
        "dataset": "simple_sim_norm",

        "runs": 10
    },
    {
        "name": "2d_cnn_v4",
        "display": "2 layers",
        "dataset": "simple_sim_norm",

        "runs": 10
    },
    {
        "name": "2d_cnn_v3",
        "display": "3 layers",
        "dataset": "simple_sim_norm",

        "runs": 10
    },
    {
        "name": "2d_cnn_v8",
        "display": "4 layers",
        "dataset": "simple_sim_norm",

        "runs": 10
    },
]

if __name__ == '__main__':
    all_loss_for_setup = []
    params_per_setup = [None for x in range(len(setups))]

    for index, setup in enumerate(setups):
        loss = []

        for run in range(setup['runs']):
            # if run == 3:
            #     continue

            data_test, ground_test, predictions, _, params = util.load_batch_setup_and_eval(setup, run)
            params_per_setup[index] = params

            loss_index_pair = util.compute_mse(predictions, ground_test)
            loss.extend([x[0] for x in loss_index_pair])

        loss = np.asarray(loss)

        # Scale MSE values to the same scale for comparison
        # prediction_results = prediction_results / (np.amax(prediction_results))

        plt.scatter(params_per_setup[index], np.mean(loss))

        all_loss_for_setup.append(loss)

    # Display neurons vs loss
    plt.title("Mean Loss VS Neuron Count")
    plt.legend([setup["name"] for setup in setups])
    plt.show()

    counts, edges, bars = plt.hist(all_loss_for_setup)

    plt.title("Prediction error distribution per setup")
    plt.legend([setup["display"] for setup in setups])
    # plt.bar_label(bars)
    plt.ylabel("Count")
    plt.xlabel("Mean Squared Error")
    plt.show()

    info = [
        (setups[i]['name'], np.mean(x), np.std(x), scipy.stats.ttest_rel(x, all_loss_for_setup[i + 1 if i < len(setups) - 1 else 0]))
        for i, x in enumerate(all_loss_for_setup)
    ]
    # info = [
    #     (setups[i]['name'], np.mean(x), np.std(x), scipy.stats.ttest_rel(x, all_loss_for_setup[-2]))
    #     for i, x in enumerate(all_loss_for_setup)
    # ]

    print(info)
