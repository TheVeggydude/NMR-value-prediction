import tensorflow as tf

# As adapted from: https://github.com/keras-team/keras/issues/10204


class ConstMultiplier(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ConstMultiplier, self).__init__(**kwargs)
        self.k = None

    def build(self, input_shape):
        self.k = self.add_weight(
            name='k',
            shape=(),
            initializer='ones',
            dtype='float32',
            trainable=True,
        )
        super(ConstMultiplier, self).build(input_shape)

    def call(self, x):
        return tf.math.multiply(self.k, x)

    def compute_output_shape(self, input_shape):
        return input_shape
