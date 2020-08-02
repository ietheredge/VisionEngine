"""
Copyright (c) 2020 R. Ian Etheredge All rights reserved.

This work is licensed under the terms of the MIT license.
For a copy, see <https://opensource.org/licenses/MIT>.
"""
from VisionEngine.base.base_model import BaseModel
from VisionEngine.layers.variational_layer import VariationalLayer
from VisionEngine.layers.perceptual_loss_layer import PerceptualLossLayer
from VisionEngine.layers.noise_layer import NoiseLayer

import tensorflow as tf


class Encoder(BaseModel):
    def __init__(self, config):
        super(Encoder, self).__init__(config)
        self.make_encoder()

    def make_encoder(self):
        with tf.name_scope('encoder'):
            self.encoder_inputs = tf.keras.layers.Input(
                shape=self.config.model.input_shape, name='input')

            if self.config.model.denoise is True:
                with tf.name_scope('noise_layer'):
                    noise_layers = tf.keras.Sequential([
                        NoiseLayer(),
                        tf.keras.layers.GaussianNoise(self.config.model.noise_ratio)
                        ], name='noise_layer')

                    noisy_inputs = noise_layers(self.encoder_inputs)

            with tf.name_scope('z_1'):
                h_1_layers = tf.keras.Sequential([
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
                    tf.keras.layers.LeakyReLU()], name='h_1')

                if self.config.modeul.denoise is True:
                    h_1 = h_1_layers(noisy_inputs)
                else:
                    h_1 = h_1_layers(self.encoder_inputs)
                h_1_flatten = tf.keras.layers.Flatten()(h_1)

            with tf.name_scope('z_2'):
                h_2_layers = tf.keras.Sequential([
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
                    tf.keras.layers.LeakyReLU()], name='h_2')

                h_2 = h_2_layers(h_1)
                h_2_flatten = tf.keras.layers.Flatten()(h_2)

            with tf.name_scope('z_3'):
                h_3_layers = tf.keras.Sequential([
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
                    tf.keras.layers.LeakyReLU()], name='h_3')

                h_3 = h_3_layers(h_2)
                h_3_flatten = tf.keras.layers.Flatten()(h_3)

            with tf.name_scope('z_4'):
                h_4_layers = tf.keras.Sequential([
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
                    tf.keras.layers.LeakyReLU()], name='h_4')

                h_4 = h_4_layers(h_3)
                h_4_flatten = tf.keras.layers.Flatten()(h_4)

            self.encoder_outputs = [
                h_1_flatten, h_2_flatten, h_3_flatten, h_4_flatten
                ]

            self.encoder = tf.keras.Model(
                self.encoder_inputs, self.encoder_outputs, name='encoder')


class Decoder(BaseModel):
    def __init__(self, config):
        super(Decoder, self).__init__(config)
        self.make_decoder()

    def make_decoder(self):
        with tf.name_scope('decoder'):

            self.z_1_input = tf.keras.layers.Input(
                (self.config.model.latent_size,), name='z_1')

            self.z_2_input = tf.keras.layers.Input(
                (self.config.model.latent_size,), name='z_2')

            self.z_3_input = tf.keras.layers.Input(
                (self.config.model.latent_size,), name='z_3')

            self.z_4_input = tf.keras.layers.Input(
                (self.config.model.latent_size,), name='z_4')

            with tf.name_scope('z_tilde_4'):
                z_4 = self.z_4_input
                z_tilde_4_layers = tf.keras.Sequential([
                    tf.keras.layers.Dense(
                        1024, kernel_regularizer=tf.keras.regularizers.l2(
                            self.config.model.kernel_l2_regularize)),

                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.ReLU(6.),
                    tf.keras.layers.Dense(
                        1024, kernel_regularizer=tf.keras.regularizers.l2(
                            self.config.model.kernel_l2_regularize)),

                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.ReLU(6.),
                    tf.keras.layers.Dense(
                        16*16*512, kernel_regularizer=tf.keras.regularizers.l2(
                            self.config.model.kernel_l2_regularize)),

                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.ReLU(6.),
                    tf.keras.layers.Reshape((16, 16, 512))], name='z_tilde_4')

                z_tilde_4 = z_tilde_4_layers(z_4)

            with tf.name_scope('z_tilde_3'):
                z_3 = tf.keras.layers.Dense(
                    16*16*512, kernel_regularizer=tf.keras.regularizers.l2(
                        self.config.model.kernel_l2_regularize))(self.z_3_input)

                z_3 = tf.keras.layers.BatchNormalization()(z_3)
                z_3 = tf.keras.layers.ReLU(6.)(z_3)
                z_3 = tf.keras.layers.Reshape((16, 16, 512))(z_3)
                z_tilde_3_layers = tf.keras.Sequential([
                    tf.keras.layers.Conv2DTranspose(
                        512, 4, 1, padding='same',
                        kernel_regularizer=tf.keras.regularizers.l2(
                            self.config.model.kernel_l2_regularize)),

                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.ReLU(6.),
                    tf.keras.layers.Conv2DTranspose(
                        256, 4, 2, padding='same',
                        kernel_regularizer=tf.keras.regularizers.l2(
                            self.config.model.kernel_l2_regularize)),

                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.ReLU(6.),
                    tf.keras.layers.Conv2DTranspose(
                        256, 4, 1, padding='same',
                        kernel_regularizer=tf.keras.regularizers.l2(
                            self.config.model.kernel_l2_regularize)),

                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.ReLU(6.)], name='z_tilde_3')

                input_z_tilde_3 = tf.keras.layers.Concatenate()([z_tilde_4, z_3])
                z_tilde_3 = z_tilde_3_layers(input_z_tilde_3)

            with tf.name_scope('z_tilde_2'):
                z_2 = tf.keras.layers.Dense(
                    32*32*256, kernel_regularizer=tf.keras.regularizers.l2(
                        2.5e-5))(self.z_2_input)
                z_2 = tf.keras.layers.BatchNormalization()(z_2)
                z_2 = tf.keras.layers.ReLU(6.)(z_2)
                z_2 = tf.keras.layers.Reshape((32, 32, 256))(z_2)
                z_tilde_2_layers = tf.keras.Sequential([
                    tf.keras.layers.Conv2DTranspose(
                        128, 4, 2, padding='same',
                        kernel_regularizer=tf.keras.regularizers.l2(
                            self.config.model.kernel_l2_regularize)),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.ReLU(6.),
                    tf.keras.layers.Conv2DTranspose(
                        128, 4, 1, padding='same',
                        kernel_regularizer=tf.keras.regularizers.l2(
                            self.config.model.kernel_l2_regularize)),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.ReLU(6.),
                    tf.keras.layers.Conv2DTranspose(
                        64, 4, 2, padding='same',
                        kernel_regularizer=tf.keras.regularizers.l2(
                            self.config.model.kernel_l2_regularize)),

                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.ReLU(6.)], name='z_tilde_2')

                input_z_tilde_2 = tf.keras.layers.Concatenate()([z_tilde_3, z_2])
                z_tilde_2 = z_tilde_2_layers(input_z_tilde_2)

            with tf.name_scope('z_tilde_1'):
                z_1 = tf.keras.layers.Dense(
                    128*128*64, kernel_regularizer=tf.keras.regularizers.l2(
                        self.config.model.kernel_l2_regularize))(self.z_1_input)

                z_1 = tf.keras.layers.BatchNormalization()(z_1)
                z_1 = tf.keras.layers.ReLU(6.)(z_1)
                z_1 = tf.keras.layers.Reshape((128, 128, 64))(z_1)
                z_tilde_1_layers = tf.keras.Sequential([
                    tf.keras.layers.Conv2DTranspose(
                        128, 4, 2, padding='same',
                        kernel_regularizer=tf.keras.regularizers.l2(
                            self.config.model.kernel_l2_regularize)),

                    tf.keras.layers.Conv2DTranspose(
                        self.config.model.input_shape[2], 4, 1, padding='same',
                        kernel_regularizer=tf.keras.regularizers.l2(
                            self.config.model.kernel_l2_regularize)),

                    tf.keras.layers.Activation('sigmoid')], name='z_tilde_1')

                input_z_tilde_1 = tf.keras.layers.Concatenate()([z_tilde_2, z_1])

                self.decoder_outputs = z_tilde_1_layers(input_z_tilde_1)
                self.decoder_inputs = [
                    self.z_1_input, self.z_2_input, self.z_3_input, self.z_4_input
                    ]

            self.decoder = tf.keras.Model(
                self.decoder_inputs,
                self.decoder_outputs,
                name='decoder')


class VAEModel(Encoder, Decoder):
    def __init__(self, config):
        super(VAEModel, self).__init__(config)
        self.make_model()

    def make_model(self):

        def weighted_reconstruction_loss(x, xhat):
            return self.config.model.recon_loss_weight \
                * tf.losses.mean_squared_error(
                    tf.keras.layers.Flatten()(x),
                    tf.keras.layers.Flatten()(xhat)) \
                * tf.cast(tf.keras.backend.prod(self.config.model.input_shape), tf.float32)

        self.inputs = tf.keras.layers.Input(self.config.model.input_shape)
        self.h_1, self.h_2, self.h_3, self.h_4 = self.encoder(self.inputs)
        with tf.name_scope('z_1_latent'):
            self.z_1 = VariationalLayer(
                size=self.config.model.latent_size,
                mu_prior=self.config.model.mu_prior,
                sigma_prior=self.config.model.sigma_prior,
                use_kl=self.config.model.use_kl,
                kl_coef=self.config.model.kl_coef,
                use_mmd=self.config.model.use_mmd,
                mmd_coef=self.config.model.mmd_coef,
                sigmas=self.config.model.sigmas,
                name='z_1_latent')(self.h_1)

        with tf.name_scope('z_2_latent'):
            self.z_2 = VariationalLayer(
                size=self.config.model.latent_size,
                mu_prior=self.config.model.mu_prior,
                sigma_prior=self.config.model.sigma_prior,
                use_kl=self.config.model.use_kl,
                kl_coef=self.config.model.kl_coef,
                use_mmd=self.config.model.use_mmd,
                mmd_coef=self.config.model.mmd_coef,
                sigmas=self.config.model.sigmas,
                name='z_2_latent')(self.h_2)

        with tf.name_scope('z_3_latent'):
            self.z_3 = VariationalLayer(
                size=self.config.model.latent_size,
                mu_prior=self.config.model.mu_prior,
                sigma_prior=self.config.model.sigma_prior,
                use_kl=self.config.model.use_kl,
                kl_coef=self.config.model.kl_coef,
                use_mmd=self.config.model.use_mmd,
                mmd_coef=self.config.model.mmd_coef,
                sigmas=self.config.model.sigmas,
                name='z_3_latent')(self.h_3)

        with tf.name_scope('z_4_latent'):
            self.z_4 = VariationalLayer(
                size=self.config.model.latent_size,
                mu_prior=self.config.model.mu_prior,
                sigma_prior=self.config.model.sigma_prior,
                use_kl=self.config.model.use_kl,
                kl_coef=self.config.model.kl_coef,
                use_mmd=self.config.model.use_mmd,
                mmd_coef=self.config.model.mmd_coef,
                sigmas=self.config.model.sigmas,
                name='z_4_latent')(self.h_4)

        self.outputs = self.decoder([self.z_1, self.z_2, self.z_3, self.z_4])
        if self.config.model.use_perceptual_loss is True:
            with tf.name_scope('perceptual_loss'):
                (_, self.outputs) = PerceptualLossLayer(
                    perceptual_loss_model=self.config.model.perceptual_loss_model,
                    pereceptual_loss_layers=self.config.model.pereceptual_loss_layers,
                    perceptual_loss_layer_weights=self.config.model.perceptual_loss_layer_weights,
                    model_input_shape=self.config.model.input_shape,
                    name='perceptual_loss_layer')([self.inputs, self.outputs])

        self.model = tf.keras.Model(self.inputs, self.outputs, name='vlae')
        self.model.summary()
        self.model.compile(tf.keras.optimizers.Adam(), weighted_reconstruction_loss)

    def load(self, checkpoint_path):
        if self.model is None:
            raise Exception("You need to build the model first.")

        print("Loading model checkpoint {} ...\n".format(checkpoint_path))
        self.model.load_weights(checkpoint_path)
        print("Model loaded")
