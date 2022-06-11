import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_model(input_shape=(301, 512)):

    inputs = tf.keras.Input(shape=input_shape)

    outputs = layers.Conv1D(
        filters=2,
        kernel_size=16,
        activation='elu',
        strides=1,
        padding='same'
    )(inputs)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.summary()

    return model


if __name__ == '__main__':
    create_model((301, 512))

