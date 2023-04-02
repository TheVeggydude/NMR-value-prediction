import numpy as np
import matplotlib.pyplot as plt


def index_to_ppm(i, ref):
    ppm = (i - ref)/ref
    
    return ppm


def plot_input(datapoint, title="", axis=0):

    plt.imshow(np.array(datapoint[:, :, axis]), interpolation='nearest')
    plt.title(title)
    plt.ylabel("Time steps")
    plt.xlabel("Frequency distribution")
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


def plot_fit_and_ground(pred, ground, title, name, comparison=None, comp_name=None):
    legend = ["Ground Pi", "Ground PCR", name + " Pi", name + " PCR"]

    if len(pred.shape) == 3:
        pred = pred.reshape(301, 2)

    if comparison is not None and len(comparison.shape) == 3:
        comparison = comparison.reshape(301, 2)

    plt.plot(ground)
    plt.plot(pred)

    if comparison is not None:
        plt.plot(comparison)
        legend.extend([comp_name + " Pi", comp_name + " PCR"])

    plt.title(title)
    plt.legend(legend)
    plt.xlabel("Time")
    plt.ylabel("Relative concentration")

    plt.show()


def plot_ground(ground, title):
    legend = ["Ground Pi", "Ground PCR"]

    plt.plot(ground)

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
