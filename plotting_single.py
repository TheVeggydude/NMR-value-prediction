import numpy as np
import matplotlib.pyplot as plt

import util


def plot_input(datapoint):

    plt.imshow(np.array(datapoint), interpolation='nearest')
    plt.show()


def plot_pcr_subset(subset, ground, title, use_legend=True):
    legend = []

    # Show each of the worst X fits
    for item in subset:

        # Plot PCr
        plt.plot(ground[item[1], :, 1])
        legend.append(
            "[" + str(item[1]) + "]: " + str(round(item[0], 4)) + " (" + format(item[2], '.1E') + ")"
        )

    plt.title(title)
    if use_legend:
        plt.legend(legend)

    plt.show()


def plot_prediction_and_ground(pred, ground, title, name, comparison=None, comp_name=None):
    legend = ["Ground Pii", "Ground PCR", name + " Pii", name + " PCR"]

    if len(pred.shape) == 3:
        pred = pred.reshape(301, 2)

    if comparison is not None and len(comparison.shape) == 3:
        comparison = comparison.reshape(301, 2)

    plt.plot(ground)
    plt.plot(pred)

    if comparison is not None:
        plt.plot(comparison)
        legend.extend([comp_name + " Pii", comp_name + " PCR"])

    plt.title(title)
    plt.legend(legend)
    plt.xlabel("Time")
    plt.ylabel("Relative concentration")

    plt.show()


def plot_mse_vs_noise(mse, noise):
    plt.scatter(mse, noise)

    plt.title("MSE score vs noise level")
    plt.xlabel("MSE")
    plt.ylabel("Noise")

    plt.show()


setup = {
        "name": "1d_cnn_v7",
        "dataset": "simple_simulation_proc",

        "batch_size": 32,
        "epochs": 10000,
        "runs": 10
    }

# Setup to compare fits with
comp_setup = {
        "name": "1d_cnn_v4",
        "dataset": "simple_simulation_proc",

        "batch_size": 32,
        "epochs": 10000,
        "runs": 10
    }

test_batch = 0


if __name__ == '__main__':

    data_test, ground_test, model, eval_score, predictions, noises = util.load_batch_setup_and_eval(setup, test_batch)
    _, _, comp_model, comp_eval_score, comp_predictions, _ = util.load_batch_setup_and_eval(comp_setup, test_batch)

    # Valuate results
    mse_results = util.compute_mse(predictions, ground_test, noises)
    sorted_mse_results = sorted(mse_results)

    # Plot best and worst fits
    plot_pcr_subset(sorted_mse_results[:10], ground_test, setup['name'] + " - Best fits")
    plot_pcr_subset(sorted_mse_results[-10:], ground_test, setup['name'] + " - Worst fits")

    # Plot best fit
    best_idx = sorted_mse_results[0][1]
    noise = util.compute_noise_metric(data_test[best_idx])

    plot_prediction_and_ground(
        predictions[best_idx],
        ground_test[best_idx],
        "Best prediction (noise = " + str(noise) + ")",
        setup['name'],
        comp_predictions[best_idx],
        comp_setup['name']
    )

    # Plot worst fit
    worst_idx = sorted_mse_results[-1][1]
    noise = util.compute_noise_metric(data_test[worst_idx])

    plot_prediction_and_ground(
        predictions[worst_idx],
        ground_test[worst_idx],
        "Worst prediction (noise = " + str(noise) + ")",
        setup['name'],
        comp_predictions[worst_idx],
        comp_setup['name']
    )

    # noise_levels = [util.compute_noise_metric(dp) for dp in data_test]
    # mse_scores = [result[0] for result in mse_results]
    #
    # plot_mse_vs_noise(mse_scores, noise_levels)
