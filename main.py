import models
import reader
import numpy as np
import scipy.fft

from sklearn.model_selection import train_test_split

# Config
directory = 'simulation/Batch 1000 2'
model_type = "MLP"
truth_shape = (2, 261)  # numpy array shape of ground truth for a single experiment.


def center_fft(x):
    return np.roll(np.abs(x), int(len(x) / 2))


def normalize_experiment(x):
    norm = np.linalg.norm(x)
    return x/norm


if __name__ == '__main__':

    print('Starting script, fetching data...')

    # DATA ACQUISITION
    data = reader.read_dat_directory(directory)
    ground_truth = reader.read_dat_directory(directory+"/ground_truth", truth_shape)
    # ground_truth = reader.read_csv(directory)

    print(data.shape)
    print(ground_truth.shape)

    data = data[:100]
    ground_truth = ground_truth[:100]

    print('Data fetches, preprocessing...')

    # PREPROCESSING
    data = np.apply_along_axis(scipy.fft.fft, 2, data)  # perform fast Fourier transform
    data = np.apply_along_axis(np.flip, 2, data)  # flip data so ATP is on the right side
    data = np.apply_along_axis(center_fft, 2, data)  # center PCR peak in the middle

    data = np.asarray([normalize_experiment(experiment) for experiment in data])  # normalize within a single experiment

    data = data.reshape(data.shape[0], data.shape[1], data.shape[2], 1)

    print('Data preprocessed, splitting...')

    # TEST, TRAIN, VALIDATION SPLIT
    X_train, X_test, y_train, y_test = train_test_split(data, ground_truth, test_size=0.33, random_state=42)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # MODELING
    print('Data split, creating model...')
    model = None

    if model_type == "MLP":
        model = models.mlp.create_model(truth_shape)
    elif model_type == "CNN":
        model = models.cnn.create_model()

    print('Model created, fitting data...')

    history = model.fit(
        X_train,
        y_train,
        batch_size=16,
        epochs=30,
        validation_data=(X_validation, y_validation),
    )

    # EVALUATION
    print("Evaluate on test data")
    results = model.evaluate(X_test, y_test, batch_size=64)
    print("test loss, test acc:", results)
