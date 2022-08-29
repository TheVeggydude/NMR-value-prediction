import random
import reader
import preprocessing
import numpy as np

dataset = "lw_90_120"
K = 10

if __name__ == '__main__':

    print("Reading data from: '" + dataset + "'")

    # Data loading & preprocessing
    data = reader.read_dat('simulation\\datasets\\' + dataset + '_data.dat', (-1, 512, 301))
    ground = reader.read_dat('simulation\\datasets\\' + dataset + '_ground_truth.dat', (-1, 2, 301))
    params = reader.read_dat('simulation\\datasets\\' + dataset + '_params.dat', (-1, 8))

    # Transpose the np arrays to the desired format
    data = np.asarray([
        preprocessing.as_absolute(  # make complex values absolute.
            np.transpose(exp)  # transpose experiments.
        ) for exp in data
    ])

    ground = np.asarray([
        preprocessing.as_absolute(  # make complex values absolute.
            np.transpose(exp)  # transpose experiments.
        ) for exp in ground
    ])

    print("Splitting dataset")

    # Track indices
    idx = [i for i in range(len(data))]

    # Randomize indices
    random.shuffle(idx)

    # Reduce idx to something dividable by K
    leftovers = []
    offset = len(idx) % K

    # Store leftover idx values, use them in the last batch
    if len(idx) % K != 0:
        leftovers = idx[len(idx) - offset:]
        idx = idx[:len(idx) - offset]

    # Divide idx into K batches
    batches = {}
    for i in range(K):
        start_pos = int(i * (len(idx)/K))
        end_pos = int((i+1) * (len(idx)/K))

        batches[i] = idx[start_pos: end_pos]

    # Add leftovers to the last batch
    batches[K-1] = batches[K-1] + leftovers

    print("Preprocessing")
    data_processed = data.copy()
    ground_processed = ground.copy()

    data_processed, ground_processed = preprocessing.preprocess(data_processed, ground_processed)

    for key, value in batches.items():

        # # Store unprocessed batches
        # np.save('simulation\\datasets\\' + dataset + "_raw_dataset_batch" + str(key), data[value])
        # np.save('simulation\\datasets\\' + dataset + "_raw_ground_truth_batch" + str(key), ground[value])
        # np.save('simulation\\datasets\\' + dataset + "_raw_params_batch" + str(key), params[value])

        # Store processed batches
        np.save('simulation\\datasets\\' + dataset + "_dataset_batch" + str(key), data_processed[value])
        np.save('simulation\\datasets\\' + dataset + "_ground_truth_batch" + str(key), ground_processed[value])
        np.save('simulation\\datasets\\' + dataset + "_params_batch" + str(key), params[value])

    # Dataset loading
    data = np.asarray([
        np.load(
            'simulation\\datasets\\' + dataset + '_dataset_batch' + str(i) + '.npy'
        ) for i in range(K)
    ])

    ground = np.asarray([
        np.load(
            'simulation\\datasets\\' + dataset + '_ground_truth_batch' + str(i) + '.npy'
        ) for i in range(K)
    ])

    params = np.asarray([
        np.load(
            'simulation\\datasets\\' + dataset + '_params_batch' + str(i) + '.npy'
        ) for i in range(K)
    ])

    # Store new, collected, numpy array
    np.save('simulation\\datasets\\' + dataset + "_dataset", data)
    np.save('simulation\\datasets\\' + dataset + "_ground_truth", ground)
    np.save('simulation\\datasets\\' + dataset + "_params", params)