from tensorflow import keras
from tensorflow.keras import layers


def create_model():

    model = keras.Sequential()
    model.add(keras.Input(shape=(261, 1000,)))  # Input is a whole experiment
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(16, activation="relu"))
    model.add(layers.Dense(1))  # Output for model, only single scalar value atm

    model.summary()

    model.compile(loss="mean_squared_error", optimizer='adam')
    return model
