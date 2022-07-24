import tensorflow as tf

# As adapted from: https://github.com/keras-team/keras/issues/10204


class VectorMultiplier(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(VectorMultiplier, self).__init__(**kwargs)
        self.k = None

    def build(self, input_shape):

        # Builds a trainable vector of length equal to the data axis of the input. This
        # vector is then used to scale the output, decoupling the scaling of the multiple
        # output variables.
        self.k = self.add_weight(
            name='k',
            shape=(input_shape[2]),
            initializer='ones',
            dtype='float32',
            trainable=True,
        )
        super(VectorMultiplier, self).build(input_shape)

    def call(self, x):

        return tf.math.multiply(self.k, x)

    def compute_output_shape(self, input_shape):
        return input_shape
