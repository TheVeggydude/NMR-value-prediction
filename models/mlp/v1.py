from tensorflow import keras
from tensorflow.keras import layers

import math


def create_model(options):
    model = keras.Sequential()

    model.add(keras.Input(shape=options['input_shape']))
    model.add(layers.Flatten())

    model.add(layers.Dense(options['n']))

    model.add(layers.Dense(math.prod(options['output_shape'])))
    model.add(layers.Reshape(options['output_shape']))

    model.summary()
    return model


if __name__ == '__main__':
    create_model(
        {
            'input_shape': (301, 512, 1),
            'output_shape': (301, 2, 1),
            'n': 6
        }
    )
