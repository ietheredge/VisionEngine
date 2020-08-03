"""
Copyright (c) 2020 R. Ian Etheredge All rights reserved.

This work is licensed under the terms of the MIT license.
For a copy, see <https://opensource.org/licenses/MIT>.
"""
from VisionEngine.base.base_model import BaseModel
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
                    tf.keras.layers.Conv2D(
                        64, 4, 2, kernel_regularizer=tf.keras.regularizers.l2(
                            self.config.model.kernel_l2_regularize),
                        kernel_initializer=tf.random_normal_initializer(
                            stddev=self.config.model.kernel_rnorm_init)),

                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.LeakyReLU(),
                    tf.keras.layers.Conv2D(
                        64, 4, 1, kernel_regularizer=tf.keras.regularizers.l2(
                            self.config.model.kernel_l2_regularize),
                        kernel_initializer=tf.random_normal_initializer(
                            stddev=self.config.model.kernel_rnorm_init)),

                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.LeakyReLU()], name='g_1')

                g_1 = g_1_layers(self.encoder_inputs)
                
                # create skip connection to variational layer
                g_1_flatten = tf.keras.layers.Flatten()(g_1)

            # hierarchical level 2
            with tf.name_scope('g_2'):

                # create the encoder block 
                g_2_layers = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(
                        128, 4, 2, kernel_regularizer=tf.keras.regularizers.l2(
                            self.config.model.kernel_l2_regularize),
                        kernel_initializer=tf.random_normal_initializer(
                            stddev=self.config.model.kernel_rnorm_init)),

                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.LeakyReLU(),
                    tf.keras.layers.Conv2D(
                        128, 4, 1, kernel_regularizer=tf.keras.regularizers.l2(
                            self.config.model.kernel_l2_regularize),
                        kernel_initializer=tf.random_normal_initializer(
                            stddev=self.config.model.kernel_rnorm_init)),

                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.LeakyReLU(),
                    tf.keras.layers.Conv2D(
                        256, 4, 2, kernel_regularizer=tf.keras.regularizers.l2(
                            self.config.model.kernel_l2_regularize),
                        kernel_initializer=tf.random_normal_initializer(
                            stddev=self.config.model.kernel_rnorm_init)),

                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.LeakyReLU()], name='g_2')

                g_2 = g_2_layers(g_1)

                # create skip connection to variational layer
                g_2_flatten = tf.keras.layers.Flatten()(g_2)

            # hierarchical level 3
            with tf.name_scope('g_3'):
    
                # create the encoder block 
                g_3_layers = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(
                        256, 4, 1, kernel_regularizer=tf.keras.regularizers.l2(
                            self.config.model.kernel_l2_regularize),
                        kernel_initializer=tf.random_normal_initializer(
                            stddev=self.config.model.kernel_rnorm_init)),

                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.LeakyReLU(),
                    tf.keras.layers.Conv2D(
                        512, 4, 2, kernel_regularizer=tf.keras.regularizers.l2(
                            self.config.model.kernel_l2_regularize),
                        kernel_initializer=tf.random_normal_initializer(
                            stddev=self.config.model.kernel_rnorm_init)),

                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.LeakyReLU(),
                    tf.keras.layers.Conv2D(
                        512, 4, 1, kernel_regularizer=tf.keras.regularizers.l2(
                            self.config.model.kernel_l2_regularize),
                        kernel_initializer=tf.random_normal_initializer(
                            stddev=self.config.model.kernel_rnorm_init)),

                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.LeakyReLU()], name='g_3')

                g_3 = g_3_layers(g_2)

                # create skip connection to variational layer
                g_3_flatten = tf.keras.layers.Flatten()(g_3)

            # highest hierarchical level
            with tf.name_scope('g_4'):

                # create the encoder block 
                g_4_layers = tf.keras.Sequential([
                    tf.keras.layers.Dense(
                        1024, kernel_regularizer=tf.keras.regularizers.l2(
                            self.config.model.kernel_l2_regularize),
                        kernel_initializer=tf.random_normal_initializer(
                            stddev=self.config.model.kernel_rnorm_init)),

                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.LeakyReLU(),
                    tf.keras.layers.Dense(
                        1024, kernel_regularizer=tf.keras.regularizers.l2(
                            self.config.model.kernel_l2_regularize),
                        kernel_initializer=tf.random_normal_initializer(
                            stddev=self.config.model.kernel_rnorm_init)),

                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.LeakyReLU()], name='g_4')

                g_4 = g_4_layers(g_3)

                # final connection to variational layer
                g_4_flatten = tf.keras.layers.Flatten()(g_4)

            self.encoder_outputs = [
                g_1_flatten, g_2_flatten, g_3_flatten, g_4_flatten
                ]

            # create the encoder model
            self.encoder = tf.keras.Model(
                self.encoder_inputs,
                self.encoder_outputs,
                name='encoder')
