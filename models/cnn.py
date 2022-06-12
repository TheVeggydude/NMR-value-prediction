from tensorflow import keras
from tensorflow.keras import layers


def create_model(input_shape=(301, 512, 1), v=1):
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

    model.summary()
    return model


if __name__ == '__main__':
    create_model((301, 512, 1))
