import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


setup = {
            "name": "2d_cnn_simple_8filters",
            "dataset": "2022-02-19T16_52_00",
            "dimensions": ((301, 2), (301, 512, 1)),
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
    mse = tf.keras.metrics.MeanSquaredError()
    mae = tf.keras.metrics.MeanAbsoluteError()
    mse_results = []

    # For each datapoint to be tested, compute the MSE.
    for index, prediction in enumerate(predictions):
        mse.reset_state()
        mse.update_state(ground_test[index], prediction)
        mse_results.append((mse.result().numpy(), index))

    # # Plot all PCR
    # for index, prediction in enumerate(predictions):
    #     plt.plot(prediction[:, 1])
    #
    # plt.title("All PCR predictions")
    # plt.show()
    #
    # # Plot all Pii
    # for index, prediction in enumerate(predictions):
    #     plt.plot(prediction[:, 0])
    #
    # plt.title("All Pii predictions")
    # plt.show()

    # Sort results
    sorted_mse_results = sorted(mse_results)

    # Show each of the worst X fits
    for result in sorted_mse_results[-10:]:
        mae.reset_state()
        mae.update_state(ground_test[result[1]], predictions[result[1]])

        # Plot PCr
        plt.plot(ground_test[result[1], :, 1])
        plt.title(setup["name"] + " PCr, MAE = " + str(np.round(mae.result().numpy(), 3)))
        plt.plot(predictions[result[1], :, 1])
        plt.legend(["ground truth", "sample"])
        plt.show()

        # Plot Pii
        plt.plot(ground_test[result[1], :, 0])
        plt.title(setup["name"] + " Pii, MAE = " + str(np.round(mae.result().numpy(), 3)))
        plt.plot(predictions[result[1], :, 0])
        plt.legend(["ground truth", "sample"])
        plt.show()

    # plt.title('Worst fits')
    # plt.show()

    # # Show each of the best X fits
    # for result in sorted_mse_results[:10]:
    #     mae.reset_state()
    #     mae.update_state(ground_test[result[1]], predictions[result[1]])
    #
    #     # Plot PCr
    #     plt.plot(ground_test[result[1], :, 1])
    #     plt.title(setup["name"] + " PCr, MAE = " + str(np.round(mae.result().numpy(), 3)))
    #     plt.plot(predictions[result[1], :, 1])
    #     plt.legend(["ground truth", "sample"])
    #     plt.show()
    #
    #     # Plot Pii
    #     plt.plot(ground_test[result[1], :, 0])
    #     plt.title(setup["name"] + " Pii, MAE = " + str(np.round(mae.result().numpy(), 3)))
    #     plt.plot(predictions[result[1], :, 0])
    #     plt.legend(["ground truth", "sample"])
    #     plt.show()

    # plt.title('Best fits')
    # plt.show()
