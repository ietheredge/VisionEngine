"""
Copyright (c) 2020 R. Ian Etheredge All rights reserved.

This work is licensed under the terms of the MIT license.
For a copy, see <https://opensource.org/licenses/MIT>.
"""
from VisionEngine.models.enocder import Encoder
from VisionEngine.models.decoder import Decoder
from VisionEngine.layers.variational_layer import VariationalLayer
from VisionEngine.layers.perceptual_loss_layer import PerceptualLossLayer
from VisionEngine.layers.noise_layer import NoiseLayer

import tensorflow as tf


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

        # denoise training
        if self.config.model.denoise is True:
            with tf.name_scope('noise_layer'):
                noise_layers = tf.keras.Sequential([
                    NoiseLayer(),
                    tf.keras.layers.GaussianNoise(self.config.model.noise_ratio)
                    ], name='noise_layer')
                noisy_inputs = noise_layers(self.inputs)

            self.h_1, self.h_2, self.h_3, self.h_4 = self.encoder(noisy_inputs)

        else:
            self.h_1, self.h_2, self.h_3, self.h_4 = self.encoder(self.inputs)

        with tf.name_scope('z_1'):
            self.z_1 = VariationalLayer(
                size=self.config.model.latent_size,
                mu_prior=self.config.model.mu_prior,
                sigma_prior=self.config.model.sigma_prior,
                use_kl=self.config.model.use_kl,
                kl_coef=self.config.model.kl_coef,
                use_mmd=self.config.model.use_mmd,
                mmd_coef=self.config.model.mmd_coef,
                sigmas=self.config.model.sigmas,
                name='z_1')(self.h_1)

        with tf.name_scope('z_2'):
            self.z_2 = VariationalLayer(
                size=self.config.model.latent_size,
                mu_prior=self.config.model.mu_prior,
                sigma_prior=self.config.model.sigma_prior,
                use_kl=self.config.model.use_kl,
                kl_coef=self.config.model.kl_coef,
                use_mmd=self.config.model.use_mmd,
                mmd_coef=self.config.model.mmd_coef,
                sigmas=self.config.model.sigmas,
                name='z_2')(self.h_2)

        with tf.name_scope('z_3'):
            self.z_3 = VariationalLayer(
                size=self.config.model.latent_size,
                mu_prior=self.config.model.mu_prior,
                sigma_prior=self.config.model.sigma_prior,
                use_kl=self.config.model.use_kl,
                kl_coef=self.config.model.kl_coef,
                use_mmd=self.config.model.use_mmd,
                mmd_coef=self.config.model.mmd_coef,
                sigmas=self.config.model.sigmas,
                name='z_3')(self.h_3)

        with tf.name_scope('z_4'):
            self.z_4 = VariationalLayer(
                size=self.config.model.latent_size,
                mu_prior=self.config.model.mu_prior,
                sigma_prior=self.config.model.sigma_prior,
                use_kl=self.config.model.use_kl,
                kl_coef=self.config.model.kl_coef,
                use_mmd=self.config.model.use_mmd,
                mmd_coef=self.config.model.mmd_coef,
                sigmas=self.config.model.sigmas,
                name='z_4')(self.h_4)

        self.outputs = self.decoder([self.z_1, self.z_2, self.z_3, self.z_4])
        if self.config.model.use_perceptual_loss is True:
            with tf.name_scope('perceptual_loss'):
                (_, self.outputs) = PerceptualLossLayer(
                    perceptual_loss_model=self.config.model.perceptual_loss_model,
                    pereceptual_loss_layers=self.config.model.pereceptual_loss_layers,
                    perceptual_loss_layer_weights=self.config.model.perceptual_loss_layer_weights,
                    model_input_shape=self.config.model.input_shape,
                    name='perceptual_loss')([self.inputs, self.outputs])

        self.model = tf.keras.Model(self.inputs, self.outputs, name='vlae')
        self.model.summary()
        self.model.compile(tf.keras.optimizers.Adam(), weighted_reconstruction_loss)

    def load(self, checkpoint_path):
        if self.model is None:
            raise Exception("You need to build the model first.")

        print("Loading model checkpoint {} ...\n".format(checkpoint_path))
        self.model.load_weights(checkpoint_path)
        print("Model loaded")
