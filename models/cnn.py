import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_model(r_shape):

    model = keras.Sequential()
    model.add(keras.Input(shape=(261, 1000, 1)))  # Input is a whole experiment, first index is batch size.
    model.add(layers.Conv2D(1, (7, 7, ), activation='relu'))
    model.add(layers.AveragePooling2D((3, 3)))
    model.add(layers.Flatten())
    # model.add(layers.Dense(10, activation="relu"))
    model.add(layers.Dense(r_shape[0] * r_shape[1]))
    model.add(layers.Reshape(r_shape))

    model.summary()

    model.compile(loss="mean_squared_error", optimizer='adam', metrics=['mean_squared_error'])
    return model
