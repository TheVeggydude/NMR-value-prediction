import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_model(options):

    inputs = tf.keras.Input(shape=options['input_shape'])

    outputs = layers.Conv1D(
        filters=2,
        kernel_size=options["kernel_size"],
        strides=1,
        padding='causal'
    )(inputs)

    outputs = layers.Activation('elu')(outputs)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.summary()

    return model


if __name__ == '__main__':

    create_model(
        {
            "input_shape": (301, 512),
            "kernel_size": 16
        }
    )
