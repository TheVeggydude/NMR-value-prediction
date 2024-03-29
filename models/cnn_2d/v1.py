from tensorflow import keras
from tensorflow.keras import layers


def create_model(options):
    model = keras.Sequential()

    model.add(keras.Input(shape=options['input_shape']))  # Input is a whole experiment, first index is batch size.

    # First Conv2D block
    model.add(
        layers.Conv2D(
            filters=1,
            kernel_size=(10, 24),
            activation='elu',
            strides=1,
            padding='same'
        )
    )

    model.add(
        layers.MaxPooling2D((1, 256))
    )

    model.summary()
    return model


if __name__ == '__main__':
    create_model(
        {
            'input_shape': (301, 512, 1)
        }
    )
