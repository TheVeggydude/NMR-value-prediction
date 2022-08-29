import matplotlib.pyplot as plt
import numpy as np


dataset = "simple_sim_norm"

param_index = 2
param_name = "Line Width"


if __name__ == '__main__':

    params = np.load('simulation\\datasets\\' + dataset + '_params.npy')
    params = np.reshape(params, (-1, 8))

    counts, edges, bars = plt.hist(params[:, param_index], bins=20)
    plt.bar_label(bars)
    plt.title("Distribution of " + param_name)
    plt.show()

    if param_index != 0:
        plt.scatter(params[:, 0], params[:, param_index])
