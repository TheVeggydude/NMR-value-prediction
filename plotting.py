import numpy
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def compute_noise_metric(datapoint):

    roi = datapoint[:, : 200]
    return np.var(roi)


def compute_mse(predictions):

    mse = tf.keras.metrics.MeanSquaredError()
    mse_results = []

    # For each datapoint to be tested, compute the MSE.
    for index, prediction in enumerate(predictions):
        mse.reset_state()
        mse.update_state(ground_test[index], prediction)
        mse_results.append((mse.result().numpy(), index))

    return mse_results


def plot_input(datapoint):

    plt.imshow(np.array(datapoint), interpolation='nearest')
    plt.show()


def plot_pcr_subset(subset, ground, title, use_legend=True):
    legend = []

    # Show each of the worst X fits
    for item in subset:

        # Plot PCr
        plt.plot(ground[item[1], :, 1])
        legend.append("idx: " + str(item[1]) + ", " + str(item[0]))

    plt.title(title)
    if use_legend:
        plt.legend(legend)

    plt.show()


def plot_mse_vs_noise(mse, noise):
    plt.scatter(mse, noise)

    plt.title("MSE score vs noise level")
    plt.xlabel("MSE")
    plt.ylabel("Noise")

    plt.show()


setup = {
            "name": "1d_cnn_v1_proc",
            "dataset": "simple_simulation_raw",
            "dimensions": ((301, 2), (301, 512)),
            "batch_size": 32,
            "runs": 10
        }

test_batch = 0


if __name__ == '__main__':

    # Load model
    model = tf.keras.models.load_model('saved_models/' + setup['name'] + '_batch' + str(test_batch))
    model.summary()

    # Load data
    data_test = np.load(
        'simulation\\datasets\\' + setup['dataset'] + '_dataset_batch' + str(test_batch) + '.npy'
    )

    ground_test = np.load(
        'simulation\\datasets\\' + setup['dataset'] + '_ground_truth_batch' + str(test_batch) + '.npy'
    )

    # Prep dataset for 2D CNN
    if len(setup['dimensions'][1]) == 3:
        data_test = np.expand_dims(data_test, 3)

    # Evaluate model
    eval_score = model.evaluate(data_test, ground_test, batch_size=setup["batch_size"])

    # Generate predictions
    predictions = model.predict(data_test)

    # Valuate results
    mse_results = compute_mse(predictions)
    sorted_mse_results = sorted(mse_results)

    # Plot best and worst fits
    plot_pcr_subset(sorted_mse_results[:10], ground_test, "Best fits")
    plot_pcr_subset(sorted_mse_results[-10:], ground_test, "Worst fits")

    # noise_levels = [compute_noise_metric(dp) for dp in data_test]
    # mse_scores = [result[0] for result in mse_results]
    #
    # plot_mse_vs_noise(mse_scores, noise_levels)
