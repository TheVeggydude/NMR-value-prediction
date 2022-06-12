from tensorflow import keras
from tensorflow.keras import layers


def create_model(input_shape):
    model = keras.Sequential()

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
    create_model((301, 512, 1))
