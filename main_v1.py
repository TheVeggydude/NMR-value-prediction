import models
import reader
import numpy as np
import preprocessing

from sklearn.model_selection import train_test_split
from tensorflow import keras

# Config
directory = 'simulation\\datasets\\Batch snr5'
model_type = "CNN"
truth_shape = (261, 2)  # numpy array shape of ground truth for a single experiment.

batch_size = 16
epochs = 10  # number of training cycles - epochs.
trials = 2  # number of runs for ML experiments.

if __name__ == '__main__':

    print('Starting script, fetching data...')

    # DATA ACQUISITION
    data = reader.read_dat_directory(directory)
    ground_truth = reader.read_dat_directory(directory+"\\ground_truth", truth_shape)

    print('Data fetched, preprocessing...')

    # PREPROCESSING
    data = preprocessing.preprocess(data, v=1)
    data = preprocessing.as_absolute(data)
    data = data.reshape(data.shape[0], data.shape[1], data.shape[2], 1)

    ground_truth = preprocessing.as_absolute(ground_truth)

    # plt.imshow(np.array(data[0]), interpolation='nearest')
    # plt.show()

    print('Data preprocessed, splitting...')

    trial_scores = []
    trial_results = []

    for i in range(trials):

        # TEST, TRAIN, VALIDATION SPLIT
        X_train, X_test, y_train, y_test = train_test_split(data, ground_truth, test_size=0.33)
        X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2)

        # MODELING
        print('Data split, creating model...')
        model = None

        if model_type == "MLP":
            model = models.mlp.mlp.create_model(truth_shape)
        elif model_type == "CNN":
            model = models.cnn.create_model(truth_shape)

        # FEATURE EXTRACTOR - FOR EVALUATION OF PERFORMANCE ONLY
        extractor = keras.Model(inputs=model.inputs,
                                outputs=[layer.output for layer in model.layers])

        print('Model created, fitting data...')

        history = model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_validation, y_validation),
        )

        # EVALUATION
        print("Evaluate on test data")
        results = model.evaluate(X_test, y_test, batch_size=batch_size)
        print("test loss, test acc:", results)

        trial_scores.append(history.history)
        trial_results.append(results[1:])

    # Compute averages
    result_matrix = np.asarray(trial_results)
    print(trial_results)

    y = np.mean(result_matrix, axis=0)  # mean per metric
    e = np.var(result_matrix, axis=0)  # variance per metric

    print(y)
    print(e)


    # # PLOTTING
    # loss_matrix = np.asarray([trial['loss'] for trial in trial_scores])
    #
    # x = np.arange(epochs)
    # y = np.mean(loss_matrix, axis=0)  # mean per epoch
    # e = np.var(loss_matrix, axis=0)  # variance per epoch
    #
    # # plt.plot(x, y)
    # plt.errorbar(x, y, e, linestyle='None', marker='^')
    # plt.show()
    #
    # result = model.predict(np.array([data[0], ]))     # Generate 1 result
    #
    # # Plot PCr
    # plt.plot(ground_truth[0, :, 0])
    # plt.title("PCr experiment 1")
    # plt.plot(result[0, :, 0])
    # plt.legend(["ground truth", "sample"])
    # plt.savefig("pcr_experiment_1.png")
    # plt.show()
    #
    # # Plot PIIn
    # plt.plot(ground_truth[0, :, 1])
    # plt.title("PIIn experiment 1")
    # plt.plot(result[0, :, 1])
    # plt.legend(["ground truth", "sample"])
    # plt.savefig("piin_experiment_1.png")
    # plt.show()
    #
    # features = extractor(np.array([data[0], ]))
    #
    # for feature in features:
    #     print(feature.shape)
    #     plt.imshow(feature[0], interpolation='nearest')
    #     plt.show()
