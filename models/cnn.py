from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf


def create_model(output_shape, input_shape=(301, 512, 1), v=1, n_rep=1):
    model = keras.Sequential()

    if v == 1:  # 1D convolutional model
        model.add(keras.Input(shape=input_shape))  # Input is a whole experiment, first index is batch size.

        model.add(
            layers.Conv1D(
                filters=256,
                kernel_size=3,
                activation='relu',
                strides=1,
                padding='same'
            )
        )
        model.add(layers.BatchNormalization())

        model.add(
            layers.Conv1D(
                filters=128,
                kernel_size=3,
                activation='relu',
                strides=1,
                padding='same'
            )
        )
        model.add(layers.BatchNormalization())

        model.add(
            layers.Conv1D(
                filters=64,
                kernel_size=3,
                activation='relu',
                strides=1,
                padding='same'
            )
        )
        model.add(layers.BatchNormalization())

        model.add(
            layers.Conv1D(
                filters=32,
                kernel_size=3,
                activation='relu',
                strides=1,
                padding='same'
            )
        )
        model.add(layers.BatchNormalization())

        model.add(
            layers.Conv1D(
                filters=16,
                kernel_size=3,
                activation='relu',
                strides=1,
                padding='same'
            )
        )
        model.add(layers.BatchNormalization())

        model.add(
            layers.Conv1D(
                filters=8,
                kernel_size=3,
                activation='relu',
                strides=1,
                padding='same'
            )
        )
        model.add(layers.BatchNormalization())

        model.add(
            layers.Conv1D(
                filters=4,
                kernel_size=3,
                activation='relu',
                strides=1,
                padding='same'
            )
        )
        model.add(layers.BatchNormalization())

        model.add(
            layers.Conv1D(
                filters=2,
                kernel_size=3,
                activation='elu',
                strides=1,
                padding='same'
            )
        )

    if v == 2:
        model.add(keras.Input(shape=input_shape))  # Input is a whole experiment, first index is batch size.

        # First Conv2D block
        model.add(
            layers.Conv2D(
                filters=64,
                kernel_size=7,
                activation='relu',
                strides=(1, 3),
                padding='same'
            )
        )
        model.add(layers.BatchNormalization())

        model.add(
            layers.Conv2D(
                filters=64,
                kernel_size=7,
                activation='relu',
                strides=(1, 3),
                padding='same'
            )
        )
        model.add(layers.BatchNormalization())

        # Second Conv2D block
        model.add(
            layers.Conv2D(
                filters=32,
                kernel_size=5,
                activation='relu',
                strides=(1, 3),
                padding='same'
            )
        )
        model.add(layers.BatchNormalization())

        model.add(
            layers.Conv2D(
                filters=16,
                kernel_size=5,
                activation='relu',
                strides=(1, 3),
                padding='same'
            )
        )
        model.add(layers.BatchNormalization())

        model.add(
            layers.Conv2D(
                filters=8,
                kernel_size=5,
                activation='relu',
                strides=(1, 3),
                padding='same'
            )
        )
        model.add(layers.BatchNormalization())

        # Third Conv2D block
        model.add(
            layers.Conv2D(
                filters=1,
                kernel_size=3,
                activation='elu',
                strides=(1, 2),
                padding='same'
            )
        )

    if v == 3:
        model.add(keras.Input(shape=input_shape))  # Input is a whole experiment, first index is batch size.

        model.add(
            layers.Conv1D(
                filters=2,
                kernel_size=4,
                activation='elu',
                strides=1,
                padding='same'
            )
        )

    if v == 4:
        model.add(keras.Input(shape=input_shape))  # Input is a whole experiment, first index is batch size.

        model.add(
            layers.Conv1D(
                filters=2,
                kernel_size=8,
                activation='elu',
                strides=1,
                padding='same'
            )
        )

    if v == 5:
        model.add(keras.Input(shape=input_shape))  # Input is a whole experiment, first index is batch size.

        model.add(
            layers.Conv1D(
                filters=2,
                kernel_size=16,
                activation='elu',
                strides=1,
                padding='same'
            )
        )

    if v == 6:
        model.add(keras.Input(shape=input_shape))  # Input is a whole experiment, first index is batch size.

        model.add(
            layers.Conv1D(
                filters=2,
                kernel_size=32,
                activation='elu',
                strides=1,
                padding='same'
            )
        )

    if v == 7:

        model.add(keras.Input(shape=input_shape))  # Input is a whole experiment, first index is batch size.

        # First Conv2D block
        model.add(
            layers.Conv2D(
                filters=1,
                kernel_size=512,
                activation='relu',
                strides=(1, 256),
                padding='same'
            )
        )

    if v == 8:
        model.add(keras.Input(shape=input_shape))  # Input is a whole experiment, first index is batch size.

        # First Conv2D block
        model.add(
            layers.Conv2D(
                filters=1,
                kernel_size=64,
                activation='relu',
                strides=(1, 32),
                padding='same'
            )
        )

        model.add(
            layers.Conv2D(
                filters=1,
                kernel_size=16,
                activation='relu',
                strides=(1, 8),
                padding='same'
            )
        )

    if v == 9:
        model.add(keras.Input(shape=input_shape))  # Input is a whole experiment, first index is batch size.

        # First Conv2D block
        model.add(
            layers.Conv2D(
                filters=8,
                kernel_size=64,
                activation='relu',
                strides=(1, 32),
                padding='same'
            )
        )

        model.add(
            layers.Conv2D(
                filters=1,
                kernel_size=16,
                activation='relu',
                strides=(1, 8),
                padding='same'
            )
        )

    if v == 10:
        model.add(keras.Input(shape=input_shape))  # Input is a whole experiment, first index is batch size.

        model.add(
            layers.Conv1D(
                filters=1,
                kernel_size=4,
                activation='elu',
                strides=1,
                padding='same'
            )
        )

    if v == 11:
        model.add(keras.Input(shape=input_shape))  # Input is a whole experiment, first index is batch size.

        # First Conv2D block
        model.add(
            layers.Conv2D(
                filters=32,
                kernel_size=64,
                activation='relu',
                strides=(1, 32),
                padding='same'
            )
        )

        model.add(
            layers.Conv2D(
                filters=1,
                kernel_size=16,
                activation='relu',
                strides=(1, 8),
                padding='same'
            )
        )

    model.summary()
    return model


if __name__ == '__main__':
    create_model((301, 2), (301, 512, 1), v=9, n_rep=1)
