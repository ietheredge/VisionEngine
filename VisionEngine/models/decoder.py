"""
Copyright (c) 2020 R. Ian Etheredge All rights reserved.

This work is licensed under the terms of the MIT license.
For a copy, see <https://opensource.org/licenses/MIT>.
"""
from VisionEngine.base.base_model import BaseModel
import tensorflow as tf


class Decoder(BaseModel):
    def __init__(self, config):
        super(Decoder, self).__init__(config)
        self.make_decoder()

    def make_decoder(self):
        with tf.name_scope('decoder'):

            # collect the decoder inputs
            self.z_tilde_1 = tf.keras.layers.Input(
                (self.config.model.latent_size,), name='z_tilde_1')
            self.z_tilde_2 = tf.keras.layers.Input(
                (self.config.model.latent_size,), name='z_tilde_2')
            self.z_tilde_3 = tf.keras.layers.Input(
                (self.config.model.latent_size,), name='z_tilde_3')
            self.z_tilde_4 = tf.keras.layers.Input(
                (self.config.model.latent_size,), name='z_tilde_4')


            # highest hierarchical level
            with tf.name_scope('f_4'):

                z_tilde_4 = self.z_tilde_4

                # create the decoder block
                f_4_layers = tf.keras.Sequential([
                    tf.keras.layers.Dense(
                        1024, kernel_regularizer=tf.keras.regularizers.l2(
                            self.config.model.kernel_l2_regularize)),

                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.ReLU(),
                    tf.keras.layers.Dense(
                        1024, kernel_regularizer=tf.keras.regularizers.l2(
                            self.config.model.kernel_l2_regularize)),

                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.ReLU(),
                    tf.keras.layers.Dense(
                        16*16*512, kernel_regularizer=tf.keras.regularizers.l2(
                            self.config.model.kernel_l2_regularize)),

                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.ReLU(),
                    tf.keras.layers.Reshape((16, 16, 512))], name='f_4')
                
                # variational input only (no upstream output)
                f_4 = f_4_layers(z_tilde_4)

            # hierarchical level 3
            with tf.name_scope('f_3'):

                # map input shapes to be equal
                z_tilde_3 = tf.keras.layers.Dense(
                    16*16*512, activation=None)(self.z_tilde_3)
                z_tilde_3 = tf.keras.layers.Reshape((16, 16, 512))(z_tilde_3)

                # create the decoder block
                f_3_layers = tf.keras.Sequential([
                    tf.keras.layers.Conv2DTranspose(
                        512, 4, 1, padding='same',
                        kernel_regularizer=tf.keras.regularizers.l2(
                            self.config.model.kernel_l2_regularize)),

                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.ReLU(),
                    tf.keras.layers.Conv2DTranspose(
                        256, 4, 2, padding='same',
                        kernel_regularizer=tf.keras.regularizers.l2(
                            self.config.model.kernel_l2_regularize)),

                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.ReLU(),
                    tf.keras.layers.Conv2DTranspose(
                        256, 4, 1, padding='same',
                        kernel_regularizer=tf.keras.regularizers.l2(
                            self.config.model.kernel_l2_regularize)),

                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.ReLU()], name='f_3')

                # combine variational input with upstream output
                f_3_input = tf.keras.layers.Concatenate()([f_4, z_tilde_3])

                f_3 = f_3_layers(f_3_input)
            
            # hierarchical level 2
            with tf.name_scope('f_2'):

                # map input shapes to be equal
                z_tilde_2 = tf.keras.layers.Dense(
                    32*32*256,activation=None)(self.z_tilde_2)
                z_tilde_2 = tf.keras.layers.Reshape((32, 32, 256))(z_tilde_2)

                # create the decoder block
                f_2_layers = tf.keras.Sequential([
                    tf.keras.layers.Conv2DTranspose(
                        128, 4, 2, padding='same',
                        kernel_regularizer=tf.keras.regularizers.l2(
                            self.config.model.kernel_l2_regularize)),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.ReLU(),
                    tf.keras.layers.Conv2DTranspose(
                        128, 4, 1, padding='same',
                        kernel_regularizer=tf.keras.regularizers.l2(
                            self.config.model.kernel_l2_regularize)),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.ReLU(),
                    tf.keras.layers.Conv2DTranspose(
                        64, 4, 2, padding='same',
                        kernel_regularizer=tf.keras.regularizers.l2(
                            self.config.model.kernel_l2_regularize)),

                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.ReLU()], name='f_2')

                # combine variational input with upstream output
                f_2_input = tf.keras.layers.Concatenate()([f_3, z_tilde_2])
            
                f_2 = f_2_layers(f_2_input)

            # lowest hierarchical level
            with tf.name_scope('f_1'):

                # map input shapes to be equal
                z_tilde_1 = tf.keras.layers.Dense(
                    128*128*64, activation=None)(self.z_tilde_1)
                z_tilde_1 = tf.keras.layers.Reshape((128, 128, 64))(z_tilde_1)

                # create the decoder block
                f_1_layers = tf.keras.Sequential([
                    tf.keras.layers.Conv2DTranspose(
                        128, 4, 2, padding='same',
                        kernel_regularizer=tf.keras.regularizers.l2(
                            self.config.model.kernel_l2_regularize)),

                    tf.keras.layers.Conv2DTranspose(
                        self.config.model.input_shape[2], 4, 1, padding='same',
                        kernel_regularizer=tf.keras.regularizers.l2(
                            self.config.model.kernel_l2_regularize)),

                    tf.keras.layers.Activation('sigmoid')], name='f_1')

                # combine variational input with upstream output
                f_1_input = tf.keras.layers.Concatenate()([f_2, z_tilde_1])

                self.decoder_outputs = f_1_layers(f_1_input)
                self.decoder_inputs = [
                    self.z_tilde_1, self.z_tilde_2, self.z_tilde_3, self.z_tilde_4
                    ]

            # create the decoder model
            self.decoder = tf.keras.Model(
                self.decoder_inputs,
                self.decoder_outputs,
                name='decoder')
