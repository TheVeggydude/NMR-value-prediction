from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf


def create_model(output_shape, input_shape=(261, 1000, 1), v=1, n_rep=1):
    model = keras.Sequential()

    if v == 1:
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
        model.add(layers.Flatten())

        model.add(layers.Dense(10, activation="relu"))
        model.add(layers.Dense(output_shape[0] * output_shape[1]))
        model.add(layers.Reshape(output_shape))

    model.summary()
    return model
