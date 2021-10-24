import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_model():

    model = keras.Sequential()
    model.add(keras.Input(shape=(261, 1000,)))  # Input is a whole experiment
    model.add(layers.Dense(500, activation="relu"))
    model.add(layers.Dense(100, activation="relu"))
    model.add(layers.Dense(1))  # Output for model, only single scalar value atm

    model.summary()

    model.compile(loss="mean_squared_error", optimizer='adam', metrics=['mean_squared_error'])
    return model
