"""
Copyright (c) 2020 R. Ian Etheredge All rights reserved.

This work is licensed under the terms of the MIT license.
For a copy, see <https://opensource.org/licenses/MIT>.
"""
from VisionEngine.base.base_model import BaseModel
from VisionEngine.layers.spectral_normalization_wrapper import (
    SpectralNormalizationWrapper,
)
import tensorflow as tf


class Decoder(BaseModel):
    def __init__(self, config):
        super(Decoder, self).__init__(config)
        self.make_decoder()

    def make_decoder(self):
        with tf.name_scope("decoder"):

            # collect the decoder inputs
            self.z_tilde_1 = tf.keras.layers.Input(
                (self.config.model.latent_size,), name="z_tilde_1"
            )
            self.z_tilde_2 = tf.keras.layers.Input(
                (self.config.model.latent_size,), name="z_tilde_2"
            )
            self.z_tilde_3 = tf.keras.layers.Input(
                (self.config.model.latent_size,), name="z_tilde_3"
            )
            self.z_tilde_4 = tf.keras.layers.Input(
                (self.config.model.latent_size,), name="z_tilde_4"
            )

            with tf.name_scope("f_4"):
                z_tilde_4 = self.z_tilde_4
                z_tilde_4 = tf.keras.layers.Dense(16 * 16 * 2048, activation=None)(
                    z_tilde_4
                )
                z_tilde_4 = tf.keras.layers.Reshape((16, 16, 2048))(z_tilde_4)

                f_4_layers = tf.keras.Sequential(
                    [
                        tf.keras.layers.UpSampling2D(),
                        SpectralNormalizationWrapper(
                            tf.keras.layers.Conv2D(2048, kernel_size=3, padding="same")
                        ),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.ReLU(),
                        SpectralNormalizationWrapper(
                            tf.keras.layers.Conv2D(1024, kernel_size=3, padding="same")
                        ),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.ReLU(),
                        SpectralNormalizationWrapper(
                            tf.keras.layers.Conv2D(1024, kernel_size=3, padding="same")
                        ),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.ReLU(),
                        SpectralNormalizationWrapper(
                            tf.keras.layers.Conv2D(512, kernel_size=3, padding="same")
                        ),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.ReLU(),
                    ],
                    name="f_4",
                )
                f_4 = f_4_layers(z_tilde_4)

            with tf.name_scope("f_3"):
                z_tilde_3 = self.z_tilde_3
                z_tilde_3 = tf.keras.layers.Dense(32 * 32 * 512, activation=None)(
                    z_tilde_3
                )
                z_tilde_3 = tf.keras.layers.Reshape((32, 32, 512))(z_tilde_3)

                f_3_layers = tf.keras.Sequential(
                    [
                        tf.keras.layers.UpSampling2D(),
                        SpectralNormalizationWrapper(
                            tf.keras.layers.Conv2D(512, kernel_size=3, padding="same")
                        ),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.ReLU(),
                        SpectralNormalizationWrapper(
                            tf.keras.layers.Conv2D(256, kernel_size=3, padding="same")
                        ),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.ReLU(),
                        SpectralNormalizationWrapper(
                            tf.keras.layers.Conv2D(256, kernel_size=3, padding="same")
                        ),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.ReLU(),
                        SpectralNormalizationWrapper(
                            tf.keras.layers.Conv2D(128, kernel_size=3, padding="same")
                        ),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.ReLU(),
                    ],
                    name="f_3",
                )

                f_3_input = tf.keras.layers.Concatenate()([f_4, z_tilde_3])

                f_3 = f_3_layers(f_3_input)

            with tf.name_scope("f_2"):
                z_tilde_2 = self.z_tilde_2
                z_tilde_2 = tf.keras.layers.Dense(64 * 64 * 128, activation=None)(
                    z_tilde_2
                )
                z_tilde_2 = tf.keras.layers.Reshape((64, 64, 128))(z_tilde_2)

                f_2_layers = tf.keras.Sequential(
                    [
                        tf.keras.layers.UpSampling2D(),
                        SpectralNormalizationWrapper(
                            tf.keras.layers.Conv2D(128, kernel_size=3, padding="same")
                        ),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.ReLU(),
                        SpectralNormalizationWrapper(
                            tf.keras.layers.Conv2D(64, kernel_size=3, padding="same")
                        ),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.ReLU(),
                        SpectralNormalizationWrapper(
                            tf.keras.layers.Conv2D(64, kernel_size=3, padding="same")
                        ),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.ReLU(),
                        SpectralNormalizationWrapper(
                            tf.keras.layers.Conv2D(32, kernel_size=3, padding="same")
                        ),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.ReLU(),
                    ],
                    name="f_2",
                )

                f_2_input = tf.keras.layers.Concatenate()([f_3, z_tilde_2])

                f_2 = f_2_layers(f_2_input)

            with tf.name_scope("f_1"):
                z_tilde_1 = self.z_tilde_1
                z_tilde_1 = tf.keras.layers.Dense(128 * 128 * 32, activation=None)(
                    z_tilde_1
                )
                z_tilde_1 = tf.keras.layers.Reshape((128, 128, 32))(z_tilde_1)

                f_1_layers = tf.keras.Sequential(
                    [
                        tf.keras.layers.UpSampling2D(),
                        SpectralNormalizationWrapper(
                            tf.keras.layers.Conv2D(32, kernel_size=3, padding="same")
                        ),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.ReLU(),
                        SpectralNormalizationWrapper(
                            tf.keras.layers.Conv2D(16, kernel_size=3, padding="same")
                        ),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.ReLU(),
                        SpectralNormalizationWrapper(
                            tf.keras.layers.Conv2D(16, kernel_size=3, padding="same")
                        ),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.ReLU(),
                        SpectralNormalizationWrapper(
                            tf.keras.layers.Conv2D(8, kernel_size=3, padding="same")
                        ),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.ReLU(),
                        tf.keras.layers.Conv2D(
                            self.config.model.input_shape[2], 3, 1, padding="same"
                        ),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.Activation("sigmoid"),
                    ],
                    name="f_1",
                )

                f_1_input = tf.keras.layers.Concatenate()([f_2, z_tilde_1])

                self.decoder_outputs = f_1_layers(f_1_input)
                self.decoder_inputs = [
                    self.z_tilde_1,
                    self.z_tilde_2,
                    self.z_tilde_3,
                    self.z_tilde_4,
                ]

            self.decoder = tf.keras.Model(
                self.decoder_inputs, self.decoder_outputs, name="decoder"
            )
