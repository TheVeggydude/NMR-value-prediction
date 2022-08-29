import numpy as np

import plotting.plotting as plot
import matplotlib.pyplot as plt
import util

from scipy import stats


setup = {
        "name": "2d_cnn_v3_k_16_8",
        "dataset": "simple_sim_norm",

        "batch_size": 32,
        "epochs": 10000,
        "runs": 10
    }

K = 10
sim_param_index = 2
sim_param_name = "Line Width"


if __name__ == '__main__':

    params = []
    mse = []

    for test_batch in range(K):

        data_test, ground_test, predictions, sim_params, _ = util.load_batch_setup_and_eval(setup, test_batch)

        # Valuate results
        mse_results = util.compute_mse(predictions, ground_test)

        params += list(sim_params[:, sim_param_index])
        mse += [result[0] for result in mse_results]

    print("Computing Spearman's correlation")
    print(stats.spearmanr(params, mse))

    print("Computing linear regression")
    reg = stats.linregress(params, mse)
    print(reg)
    plt.plot(params, reg.intercept + reg.slope * np.asarray(params), 'r')

    # x = np.arange(5, 41, 0.2)
    # y = 1 / (x - 4.9) + 0.2
    #
    # plt.plot(x, y, 'r')

    plot.plot_scatter(params, mse, setup["name"] + " MSE score vs " + sim_param_name, sim_param_name, "MSE")


