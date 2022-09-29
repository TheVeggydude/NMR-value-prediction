import random
import reader
import preprocessing
import numpy as np

dataset = "imag"
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

    data_batched = np.asarray([
        data_processed[value] for _, value in batches.items()
    ])

    ground_batched = np.asarray([
        ground_processed[value] for _, value in batches.items()
    ])

    params_batched = np.asarray([
        params[value] for _, value in batches.items()
    ])

    # Store new, collected, numpy array
    np.save('simulation\\datasets\\' + dataset + "_abs_dataset", data_batched)
    np.save('simulation\\datasets\\' + dataset + "_abs_ground_truth", ground_batched)
    np.save('simulation\\datasets\\' + dataset + "_abs_params", params_batched)
