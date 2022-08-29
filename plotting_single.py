import plotting.plotting as plot
from operator import itemgetter

import util


setup = {
        "name": "2d_cnn_v1",
        "display": "1 layer",
        "dataset": "simple_sim_norm",
    }


# Setup to compare fits with
comp_setup = {
        "name": "2d_cnn_v3",
        "display": "3 layer",
        "dataset": "simple_sim_norm",
    }

test_batch = 0
param = 1


if __name__ == '__main__':

    data_test, ground_test, predictions, sim_params, _ = util.load_batch_setup_and_eval(setup, test_batch)
    _, _, comp_predictions, _, _ = util.load_batch_setup_and_eval(comp_setup, test_batch)

    # Valuate results
    mse_results = util.compute_mse(predictions, ground_test)
    mse_results = [(x[0], x[1], sim_params[x[1], param]) for x in mse_results]
    mse_results.sort(key=itemgetter(0))

    # Plot best and worst fits
    plot.plot_pcr_subset(mse_results[:10], ground_test, setup['display'] + " - Best fits")
    plot.plot_pcr_subset(mse_results[-10:], ground_test, setup['display'] + " - Worst fits")

    # Plot best fit
    score, idx, _ = mse_results[0]
    # plot.plot_input(data_test[idx], "Best")

    plot.plot_ground_pcr_pii(
        predictions[idx],
        ground_test[idx],
        # "Best (" + str(score) + ", " + str(sim_params[idx][0]) + ")",
        "Best (" + str(score) + ")",
        setup['display'],
        comp_predictions[idx],
        comp_setup['display']
    )

    # Plot worst fit
    score, idx, _ = mse_results[-1]
    # plot.plot_input(data_test[idx], "Worst")

    plot.plot_ground_pcr_pii(
        predictions[idx],
        ground_test[idx],
        # "Worst (" + str(score) + ", " + str(sim_params[idx][0]) + ")",
        "Worst (" + str(score) + ")",
        setup['display'],
        comp_predictions[idx],
        comp_setup['display']
    )

    # mse_results.sort(key=itemgetter(2))
    #
    # score, idx, _ = mse_results[0]
    # plot.plot_input(data_test[idx], str(sim_params[idx][0:3]))
    #
    # plot.plot_prediction_and_ground(
    #     predictions[idx],
    #     ground_test[idx],
    #     "Best (" + str(score) + ", " + str(sim_params[idx][0]) + ")",
    #     setup['name'],
    #     # comp_predictions[idx],
    #     # comp_setup['name']
    # )
    #
    # # Plot worst fit
    # score, idx, _ = mse_results[-1]
    # plot.plot_input(data_test[idx], str(sim_params[idx][0:3]))
    #
    # plot.plot_prediction_and_ground(
    #     predictions[idx],
    #     ground_test[idx],
    #     "Worst (" + str(score) + ", " + str(sim_params[idx][0]) + ")",
    #     setup['name'],
    #     # comp_predictions[idx],
    #     # comp_setup['name']
    # )
