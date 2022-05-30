import reader
import preprocessing
import models
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split


# ISODate = "2022-02-19T16_52_00"
# batch_size = 64
# epochs = 50


def create_and_train_model(ISODate, batch_size, epochs):
    print("Acquiring data...")

    # Data loading & preprocessing
    data = reader.read_dat('simulation\\datasets\\' + ISODate + '_dataset.dat', (-1, 512, 301))
    data = preprocessing.preprocess(data)
    # data = data.reshape(data.shape[0], data.shape[1], data.shape[2], 1)

    ground = reader.read_dat('simulation\\datasets\\' + ISODate + '_ground_truth.dat', (-1, 2, 301))
    ground = preprocessing.preprocess(ground, transpose=False)

    print("Data acquired, splitting...")

    # Test, train, validation split
    X_train, X_test, y_train, y_test = train_test_split(data, ground, test_size=0.33)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2)

    # Model definition
    model = models.cnn.create_model((2, 301), (301, 512))

    # Training
    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_validation, y_validation),
    )

    # Testing
    print("Evaluate on test data")
    results = model.evaluate(X_test, y_test, batch_size=batch_size)
    print("test loss, test acc:", results)


if __name__ == '__main__':

    setups = [
        # {
        #     "name": "1D CNN (simple)",
        #     "ISODate": "2022-02-19T16_52_00",
        #     "type": "cnn",
        #     "v": 6,
        #     "metabolites": 2,
        #     "dimensions": ((301, 2), (301, 512)),  # (output_shape, input_shape)
        #     "n_rep": 1,
        #     "batch_size": 32,
        #     "epochs": 10000,
        #     "runs": 10,
        # },
        # {
        #     "name": "Conv2D (2 layers, 1 filter)",
        #     "ISODate": "2022-02-19T16_52_00",
        #     "type": "cnn",
        #     "v": 8,
        #     "metabolites": 2,
        #     "dimensions": ((301, 2), (301, 512, 1)),
        #     "n_rep": 6,
        #     "batch_size": 32,
        #     "epochs": 10000,
        #     "runs": 10,
        # },
        {
            "name": "Conv2D (2 layers, 8 filters)",
            "ISODate": "2022-02-19T16_52_00",
            "type": "cnn",
            "v": 9,
            "metabolites": 2,
            "dimensions": ((301, 2), (301, 512, 1)),
            "n_rep": 1,
            "batch_size": 32,
            "epochs": 10000,
            "runs": 10,
        },
        {
            "name": "Conv2D (2 layers, 32 filters)",
            "ISODate": "2022-02-19T16_52_00",
            "type": "cnn",
            "v": 11,
            "metabolites": 2,
            "dimensions": ((301, 2), (301, 512, 1)),
            "n_rep": 1,
            "batch_size": 32,
            "epochs": 10000,
            "runs": 10,
        },
        # {
        #     "name": "MLP",
        #     "ISODate": "2022-02-19T16_52_00",
        #     "type": "mlp",
        #     "v": 2,
        #     "dimensions": ((301, 2), (301, 512, 1)),  # (output_shape, input_shape)
        #     "n_rep": 1,
        #     "batch_size": 32,
        #     "epochs": 10000,
        #     "runs": 10,
        # }
    ]

    all_predictions_for_setup = []

    for setup in setups:

        experiment_results = []
        experiment_acc = []

        prediction_results = []

        print("Acquiring data...")

        # Data loading & preprocessing
        data = reader.read_dat('simulation\\datasets\\' + setup["ISODate"] + '_dataset.dat', (-1, 512, 301))
        ground = reader.read_dat('simulation\\datasets\\' + setup["ISODate"] + '_ground_truth.dat', (-1, 2, 301))

        data, ground = preprocessing.preprocess(data, ground)

        if setup["metabolites"] == 1:
            ground = ground[:, :, 1]

            if setup["type"] == "cnn":
                ground = ground.reshape(ground.shape[0], ground.shape[1], 1)

        print(ground.shape)

        if len(data.shape)-1 != len(setup['dimensions'][1]):
            # Reshape data to fit Conv2D's expected shape
            data = data.reshape(data.shape[0], data.shape[1], data.shape[2], 1)

        print(data.shape)

        # Per experimental setup, run the model some number of times.
        for run in range(setup["runs"]):

            print("Data acquired, splitting...")

            # Test, train, validation split
            X_train, X_test, y_train, y_test = train_test_split(data, ground, test_size=0.33)
            X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2)

            # Model definition
            model = None

            if setup["type"] == "mlp":
                model = models.mlp
            elif setup["type"] == "cnn":
                model = models.cnn
            elif setup["type"] == "hybrid":
                model = models.hybrid

            model = model.create_model(
                setup["dimensions"][0],
                setup["dimensions"][1],
                v=setup["v"],
                n_rep=setup['n_rep']
            )

            model.compile(
                loss="mean_squared_error",
                optimizer='adam',
                metrics=[
                    tf.keras.metrics.RootMeanSquaredError(),
                    tf.keras.metrics.MeanAbsoluteError(),
                    tf.keras.metrics.MeanAbsolutePercentageError()
                ]
            )

            # Training
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )

            history = model.fit(
                X_train,
                y_train,
                batch_size=setup["batch_size"],
                epochs=setup["epochs"],
                validation_data=(X_validation, y_validation),
                callbacks=[early_stopping]
            )

            # Testing
            print("Evaluate on test data")
            eval = model.evaluate(X_test, y_test, batch_size=setup["batch_size"])
            print("test loss, test acc:", eval)

            experiment_results.append(history.history)
            experiment_acc.append(eval[1:])

            predictions = model.predict(X_test)
            mse = tf.keras.metrics.MeanSquaredError()
            mae = tf.keras.metrics.MeanAbsoluteError()
            mse_results = []

            # For each datapoint to be tested, compute the MSE.
            for index, prediction in enumerate(predictions):
                mse.reset_state()
                mse.update_state(y_test[index], prediction)
                mse_results.append((mse.result().numpy(), index))

            #     plt.plot(prediction[:, 1])
            #
            # plt.title("All predictions")
            # plt.show()

            prediction_results.extend([x[0] for x in mse_results])

            # # Sort results
            # sorted_mse_results = sorted(mse_results)
            #
            # # Inspect best and worst fit
            # for index in [sorted_mse_results[0][1], sorted_mse_results[-1][1]]:
            #
            #     mae.reset_state()
            #     mae.update_state(y_test[index], predictions[index])
            #
            #     # Plot PCr
            #     plt.plot(y_test[index, :, 1])
            #     plt.title(setup["name"] + " PCr, MAE = " + str(np.round(mae.result().numpy(), 3)))
            #     plt.plot(predictions[index, :, 1])
            #     plt.legend(["ground truth", "sample"])
            #     plt.savefig("pcr_experiment_1.png")
            #     plt.show()

                # # Plot PIIn
                # plt.plot(ground[0, :, 0])
                # plt.title("PIIn experiment 1")
                # plt.plot(result[0, :, 0])
                # plt.legend(["ground truth", "sample"])
                # plt.savefig("piin_experiment_1.png")
                # plt.show()

        all_predictions_for_setup.append(prediction_results)

    plt.hist(all_predictions_for_setup)

    plt.title("Prediction error distribution per setup")
    plt.legend([setup["name"] for setup in setups])
    plt.ylabel("Count")
    plt.xlabel("Mean Squared Error")
    plt.show()

    #     result_matrix = np.asarray(experiment_acc)
    #
    #     loss_matrix = np.asarray([np.asarray(exp['loss']) for exp in experiment_results])
    #     print(loss_matrix.shape)
    #
    #     x = np.arange(setup["epochs"])
    #     y = np.mean(loss_matrix, axis=0)  # mean per epoch
    #     # e = np.var(loss_matrix, axis=0)  # variance per epoch
    #     e = np.zeros(setup["epochs"])
    #
    #     # plt.plot(x, y)
    #     plt.errorbar(x, y, e, linestyle='None', marker='^')
    #
    # plt.title("Comparison of mean")
    # plt.legend([setup["name"] for setup in setups])
    # plt.show()
