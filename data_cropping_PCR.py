import numpy as np

dataset = "simple_simulation_proc"
K = 10


def crop_batch(batch):
    pass


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
    data = data[:, :, :, 240:270]
    ground = ground[:, :, :, 1]

    # Store unprocessed batches
    np.save('simulation\\datasets\\' + dataset + "_PCR_dataset", data)
    np.save('simulation\\datasets\\' + dataset + "_PCR_ground_truth", ground)
