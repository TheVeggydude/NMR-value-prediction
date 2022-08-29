from tensorflow import keras
from tensorflow.keras import layers


def create_model(options):
    model = keras.Sequential()

    model.add(keras.Input(shape=options['input_shape']))  # Input is a whole experiment, first index is batch size.

    # First Conv2D block
    model.add(
        layers.Conv2D(
            filters=options['filters'][0],
            kernel_size=options['kernels'][0],
            strides=1,
            padding='same',
            activation='elu'
        )
    )

    model.add(layers.BatchNormalization())

    model.add(
        layers.MaxPooling2D((1, 8))
    )

    # Second Conv2D block
    model.add(
        layers.Conv2D(
            filters=options['filters'][1],
            kernel_size=options['kernels'][1],
            strides=1,
            padding='same',
            activation='elu'
        )
    )

    model.add(layers.BatchNormalization())

    model.add(
        layers.MaxPooling2D((1, 8))
    )

    # Third Conv2D block
    model.add(
        layers.Conv2D(
            filters=options['filters'][2],
            kernel_size=options['kernels'][2],
            strides=1,
            padding='same',
            activation='elu'
        )
    )

    model.add(layers.BatchNormalization())

    model.add(
        layers.MaxPooling2D((1, 4))
    )

    # Fourth Conv2D block
    model.add(
        layers.Conv2D(
            filters=1,
            kernel_size=options['kernels'][3],
            strides=1,
            padding='same',
            activation='elu'
        )
    )

    model.summary()
    return model


if __name__ == '__main__':

    create_model(
        {
            'input_shape': (301, 512, 1),
            'filters': [
                32,
                16,
                8
            ],
            'kernels': [
                (10, 24),
                (10, 10),
                (10, 2),
                (10, 2),
            ]
        }
    )

    # First version of v3 - DO NOT REMOVE
    # create_model(
    #     {
    #         'input_shape': (301, 512, 1),
    #         'filters': [
    #             32,
    #             16
    #         ],
    #         'kernels': [
    #             (20, 20),
    #             (10, 10),
    #             (10, 10),
    #         ]
    #     }
    # )
