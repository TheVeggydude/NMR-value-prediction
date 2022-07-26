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
            filters=options['filters'],
            kernel_size=(16, options['input_shape'][1]),
            strides=(1, options['input_shape'][1]),
            padding='same'
        )
    )

    model.add(layers.Reshape(options['output_shape']))

    model.add(layers.Activation('elu'))

    model.summary()
    return model


if __name__ == '__main__':
    create_model(
        {
            'filters': 2,
            'input_shape': (301, 55, 1),
            'output_shape': (301, 2, 1),
        }
    )
