import plotting.plotting as plot

import util


setup = {
        "name": "2d_cnn_v3_f_64_32",
        "dataset": "simple_simulation_proc",

        "batch_size": 32,
        "epochs": 10000,
        "runs": 10
    }

# Setup to compare fits with
comp_setup = {
        "name": "2d_cnn_v3_f_16_8",
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
    plot.plot_pcr_subset(sorted_mse_results[:10], ground_test, setup['name'] + " - Best fits")
    plot.plot_pcr_subset(sorted_mse_results[-10:], ground_test, setup['name'] + " - Worst fits")

    # Plot best fit
    best_idx = sorted_mse_results[0][1]
    noise = util.compute_noise_metric(data_test[best_idx])

    plot.plot_prediction_and_ground(
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

    plot.plot_prediction_and_ground(
        predictions[worst_idx],
        ground_test[worst_idx],
        "Worst prediction (noise = " + str(noise) + ")",
        setup['name'],
        comp_predictions[worst_idx],
        comp_setup['name']
    )

    noise_levels = [util.compute_noise_metric(dp) for dp in data_test]
    mse_scores = [result[0] for result in mse_results]

    plot.plot_mse_vs_noise(mse_scores, noise_levels)
