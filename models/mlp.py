from tensorflow import keras
from tensorflow.keras import layers


def create_model(r_shape):

    # V1.0
    # model = keras.Sequential()
    # model.add(keras.Input(shape=(261, 1000, 1)))  # Input is a whole experiment
    # model.add(layers.Dense(32, activation="relu"))
    # model.add(layers.Dense(16, activation="relu"))
    # model.add(layers.Dense(8, activation="relu"))
    # model.add(layers.Flatten())
    # model.add(layers.Dense(1))  # Output for model, only single scalar value atm

    # V2.0
    model = keras.Sequential()
    model.add(keras.Input(shape=(261, 1000, 1)))  # Input is a whole experiment
    model.add(layers.Flatten())
    # model.add(layers.Dense(r_shape[0] * r_shape[1] * 4, activation="relu"))
    # model.add(layers.Dense(r_shape[0] * r_shape[1] * 2, activation="relu"))
    model.add(layers.Dense(r_shape[0] * r_shape[1], activation="relu"))
    model.add(layers.Reshape(r_shape))

    model.summary()

    model.compile(loss="mean_squared_error", optimizer='adam', metrics=['mean_squared_error'])
    return model
