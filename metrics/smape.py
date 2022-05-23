# Import Keras backend
import keras.backend as k


# Define SMAPE loss function
def smape(true, predicted):
    epsilon = 0.1

    summ = k.maximum(k.abs(true) + k.abs(predicted) + epsilon, 0.5 + epsilon)
    smape_result = k.abs(predicted - true) / summ * 2.0
    return smape_result
