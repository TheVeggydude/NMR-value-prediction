import numpy as np
import matplotlib.pyplot as plt


def plot_input(datapoint, title=""):

    plt.imshow(np.array(datapoint), interpolation='nearest')
    plt.title(title)
    plt.show()


def plot_pcr_subset(subset, ground, title, use_legend=True):
    legend = []

    # Show each of the worst X fits
    for item in subset:

        # Plot PCr
        plt.plot(ground[item[1], :, 1])
        legend.append(
            "[" + str(item[1]) + "]: " + str(round(item[0], 4))
        )

    plt.title(title)
    if use_legend:
        plt.legend(legend)

    plt.show()


def plot_ground_pcr_pii(pred, ground, title, name, comparison=None, comp_name=None):
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


def plot_scatter(x, y, title="", x_label="", y_label=""):
    """
    Plots a matplotlib.pyplot.scatter using the provided data.

        Parameters:
            x ([number]): list of numbers to be plotted along the x-axis
            y ([number]): list of numbers to be plotted along the y-axis
            title (str): string for the title of the plot
            x_label (str): string for the x-axis unit
            y_label (str): string for the y-axis unit

        Returns:
            None
    """

    plt.scatter(x, y)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.show()
