import matplotlib.pyplot as plt
import numpy as np

dataset = "simple_simulation_proc"
K = 10


if __name__ == '__main__':

    # Load dataset
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

    # Crop the dataset
    data = data[:, :, :, 215:270]

    # Store unprocessed batches
    np.save('simulation\\datasets\\' + dataset + "_cropped_dataset", data)
    np.save('simulation\\datasets\\' + dataset + "_cropped_ground_truth", ground)
