import numpy as np
import matplotlib.pyplot as plt


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