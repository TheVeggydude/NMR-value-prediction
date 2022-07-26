import numpy as np

dataset = "simple_simulation_raw"
K = 10

if __name__ == '__main__':
    """
    Collects the batches of the dataset into a single, larger .npy file. This is much faster to load and easier to 
    archive.
    """

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

    # Store new, collected, numpy array
    np.save('simulation\\datasets\\' + dataset + "_dataset", data)
    np.save('simulation\\datasets\\' + dataset + "_ground_truth", ground)

