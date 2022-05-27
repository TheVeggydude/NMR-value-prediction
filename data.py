import random
import reader
import preprocessing
import numpy as np

dataset = "2022-02-19T16_52_00"
K = 10

if __name__ == '__main__':

    print("Reading data from: '" + dataset + "'")

    # Data loading & preprocessing
    data = reader.read_dat('simulation\\datasets\\' + dataset + '_dataset.dat', (-1, 512, 301))
    ground = reader.read_dat('simulation\\datasets\\' + dataset + '_ground_truth.dat', (-1, 2, 301))

    print("Preprocessing")

    data, ground = preprocessing.preprocess(data, ground)

    print("Splitting dataset")

    # Track indices
    idx = [i for i in range(len(data))]
    # idx = idx[:len(idx)-1]

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

    for key, value in batches.items():
        np.save('simulation\\datasets\\' + dataset + "_dataset_batch" + str(key), data[value])
        np.save('simulation\\datasets\\' + dataset + "_ground_truth_batch" + str(key), ground[value])
