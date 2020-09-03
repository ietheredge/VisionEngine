"""
Copyright (c) 2020 R. Ian Etheredge All rights reserved.

This work is licensed under the terms of the MIT license.
For a copy, see <https://opensource.org/licenses/MIT>.
"""
import tensorflow as tf


class SqueezeExciteLayer(tf.keras.layers.Layer):
    def __init__(self, c, r=16, **kwargs):
        super(SqueezeExciteLayer, self).__init__(**kwargs)
        self.c = c
        self.r = r

    def build(self, input_shape):
        self.se = tf.keras.Sequential(
            [
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(self.c // self.r, use_bias=False),
                tf.keras.layers.Activation("relu"),
                tf.keras.layers.Dense(self.c, use_bias=False),
                tf.keras.layers.Activation("sigmoid"),
            ]
        )  # tanh/relu worsens performance

    def call(self, layer_inputs, **kwargs):
        return self.se(layer_inputs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {"c": self.c, "r": self.r}
        base_config = super(SqueezeExciteLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
