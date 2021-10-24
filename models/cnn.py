import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_model():

    model = keras.Sequential()
    model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(261, 1000,)))
    model.add(layers.AveragePooling2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten)
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(16, activation="relu"))
    model.add(layers.Dense(8, activation="relu"))
    model.add(layers.Dense(1))  # Output for model, only single scalar value atm

    model.summary()

    model.compile(loss="mean_squared_error", optimizer='adam', metrics=['mean_squared_error'])
    return model
