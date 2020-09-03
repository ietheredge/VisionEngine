"""
Copyright (c) 2020 R. Ian Etheredge All rights reserved.

This work is licensed under the terms of the MIT license.
For a copy, see <https://opensource.org/licenses/MIT>.
"""
import tensorflow as tf


class NoiseLayer(tf.keras.layers.Layer):
    def __init__(self, ratio=0.9, **kwargs):
        super(NoiseLayer, self).__init__(**kwargs)
        self.masking = True
        self.ratio = ratio

    def call(self, inputs, training=None):
        def noised():
            shp = tf.keras.backend.shape(inputs)[1:]
            mask_select = tf.keras.backend.random_binomial(shape=shp, p=self.ratio)

            mask_noise = tf.keras.backend.random_binomial(shape=shp, p=0.1)
            out = (inputs * (mask_select)) + mask_noise
            return out

        return tf.keras.backend.in_train_phase(noised, inputs, training=training)

    def get_config(self):
        config = {"ratio": self.ratio, "masking": self.masking}
        base_config = super(NoiseLayer, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
