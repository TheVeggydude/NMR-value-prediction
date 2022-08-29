from tensorflow import keras
from tensorflow.keras import layers


###
# Model for PCR only
###


def create_model(options):
    model = keras.Sequential()

    model.add(keras.Input(shape=options['input_shape']))  # Input is a whole experiment, first index is batch size.

    # First Conv2D block
    model.add(
        layers.Conv2D(
            filters=options['filters'][0],
            kernel_size=options['kernels'][0],
            strides=1,
            padding='same'
        )
    )

    model.add(layers.Activation('elu'))

    model.add(
        layers.MaxPooling2D((1, 256))
    )

    # Second Conv2D block
    model.add(
        layers.Conv2D(
            filters=options['filters'][1],
            kernel_size=options['kernels'][1],
            strides=1,
            padding='same'
        )
    )

    model.add(layers.Activation('elu'))

    model.summary()
    return model


if __name__ == '__main__':
    create_model(
        {
            'input_shape': (301, 512, 1),
            'filters': [
                256,
                1,
            ],
            'kernels': [
                (10, 24),
                (10, 2),
            ]
        }
    )
