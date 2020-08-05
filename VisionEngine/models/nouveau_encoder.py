"""
Copyright (c) 2020 R. Ian Etheredge All rights reserved.

This work is licensed under the terms of the MIT license.
For a copy, see <https://opensource.org/licenses/MIT>.
"""
from VisionEngine.base.base_model import BaseModel
from VisionEngine.layers.spectral_normalization_wrapper import SpectralNormalizationWrapper
from VisionEngine.layers.squeeze_excite_layer import SqueezeExciteLayer
import tensorflow as tf


class Encoder(BaseModel):
    def __init__(self, config):
        super(Encoder, self).__init__(config)
        self.make_encoder()

    def make_encoder(self):
        with tf.name_scope('encoder'):

            self.encoder_inputs = tf.keras.layers.Input(
                shape=self.config.model.input_shape, name='input')

            # lowest hierarchical level
            with tf.name_scope('g_1'):

                # create the encoder block
                g_1_layers = tf.keras.Sequential([
                    tf.keras.layers.Input(self.config.model.input_shape),
                    SpectralNormalizationWrapper(tf.keras.layers.Conv2D(
                        8, 3,  padding='same')),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.LeakyReLU(),
                    SpectralNormalizationWrapper(tf.keras.layers.Conv2D(
                        16, 3,  padding='same')),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.LeakyReLU(),
                    SpectralNormalizationWrapper(tf.keras.layers.Conv2D(
                        16, 3, padding='same')),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.LeakyReLU(),
                    SpectralNormalizationWrapper(tf.keras.layers.Conv2D(
                        32, 3, padding='same')),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.LeakyReLU(),
                    tf.keras.layers.AveragePooling2D()], name='g_1')

                g_1 = g_1_layers(self.encoder_inputs)

                g_1_flatten = SqueezeExciteLayer(c=64)(g_1)

            with tf.name_scope('g_2'):
                g_2_layers = tf.keras.Sequential([
                    SpectralNormalizationWrapper(tf.keras.layers.Conv2D(
                        32, 3, padding='same')),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.LeakyReLU(),
                    SpectralNormalizationWrapper(tf.keras.layers.Conv2D(
                        64, 3, padding='same')),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.LeakyReLU(),
                    SpectralNormalizationWrapper(tf.keras.layers.Conv2D(
                        64, 3, padding='same')),
                    tf.keras.layers.LeakyReLU(),
                    SpectralNormalizationWrapper(tf.keras.layers.Conv2D(
                        128, 3, padding='same')),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.LeakyReLU(),
                    tf.keras.layers.AveragePooling2D(),
                ], name='g_2')

                g_2 = g_2_layers(g_1)
                g_2_flatten = SqueezeExciteLayer(c=128)(g_2)

            with tf.name_scope('g_3'):
                g_3_layers = tf.keras.Sequential([
                    SpectralNormalizationWrapper(tf.keras.layers.Conv2D(
                        128, 3, padding='same')),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.LeakyReLU(),
                    SpectralNormalizationWrapper(tf.keras.layers.Conv2D(
                        256, 3, padding='same')),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.LeakyReLU(),
                    SpectralNormalizationWrapper(tf.keras.layers.Conv2D(
                        256, 3, padding='same')),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.LeakyReLU(),
                    SpectralNormalizationWrapper(tf.keras.layers.Conv2D(
                        512, 3, padding='same')),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.LeakyReLU(),
                    tf.keras.layers.AveragePooling2D(),
                ], name='g_3')

                g_3 = g_3_layers(g_2)
                g_3_flatten = SqueezeExciteLayer(c=256)(g_3)

            with tf.name_scope('g_4'):
                g_4_layers = tf.keras.Sequential([
                    SpectralNormalizationWrapper(tf.keras.layers.Conv2D(
                        512, 3, padding='same')),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.LeakyReLU(),
                    SpectralNormalizationWrapper(tf.keras.layers.Conv2D(
                        1024, 3, padding='same')),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.LeakyReLU(),
                    SpectralNormalizationWrapper(tf.keras.layers.Conv2D(
                        1024, 3, padding='same')),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.LeakyReLU(),
                    SpectralNormalizationWrapper(tf.keras.layers.Conv2D(
                        2048, 3, padding='same')),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.LeakyReLU(),
                    tf.keras.layers.AveragePooling2D()], name='g_4')

                g_4 = g_4_layers(g_3)
                g_4_flatten = SqueezeExciteLayer(c=512)(g_4)

            self.encoder_outputs = [
                g_1_flatten, g_2_flatten, g_3_flatten, g_4_flatten
                ]

            self.encoder = tf.keras.Model(
                self.encoder_inputs, self.encoder_outputs, name='encoder')