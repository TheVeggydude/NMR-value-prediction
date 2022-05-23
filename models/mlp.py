from tensorflow import keras
from tensorflow.keras import layers


def create_model(output_shape, input_shape=(261, 1000, 1), v=1, n_rep=1):
    model = keras.Sequential()

    if v == 1:
        model.add(keras.Input(shape=input_shape))  # Input is a whole experiment
        model.add(layers.Flatten())
        model.add(layers.Dense(output_shape[0] * output_shape[1]))
        model.add(layers.Reshape(output_shape))

    if v == 2:
        model.add(keras.Input(shape=input_shape))  # Input is a whole experiment
        model.add(layers.Flatten())
        model.add(layers.Dense(10, activation='relu'))
        model.add(layers.Dense(output_shape[0] * output_shape[1]))
        model.add(layers.Reshape(output_shape))

    if v == 3:
        model.add(keras.Input(shape=input_shape))  # Input is a whole experiment
        model.add(layers.Flatten())
        model.add(layers.Dense(100, activation='relu'))
        model.add(layers.Dense(10, activation='relu'))
        model.add(layers.Dense(output_shape[0] * output_shape[1]))
        model.add(layers.Reshape(output_shape))

    if v == 4:
        model.add(keras.Input(shape=input_shape))  # Input is a whole experiment
        model.add(layers.Flatten())
        model.add(layers.Dense(1000, activation='relu'))
        model.add(layers.Dense(100, activation='relu'))
        model.add(layers.Dense(10, activation='relu'))
        model.add(layers.Dense(output_shape[0] * output_shape[1]))
        model.add(layers.Reshape(output_shape))

    model.summary()
    return model


if __name__ == '__main__':
    create_model((301, 2), (301, 512, 1), v=2, n_rep=1)
