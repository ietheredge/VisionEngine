"""
Copyright (c) 2020 R. Ian Etheredge All rights reserved.

This work is licensed under the terms of the MIT license.
For a copy, see <https://opensource.org/licenses/MIT>.
"""
from VisionEngine.models.encoder import Encoder
from VisionEngine.models.decoder import Decoder
from VisionEngine.layers.variational_layer import VariationalLayer
from VisionEngine.layers.perceptual_loss_layer import PerceptualLossLayer
from VisionEngine.layers.noise_layer import NoiseLayer

import tensorflow as tf


class VLAEModel(Encoder, Decoder):
    def __init__(self, config):
        super(VLAEModel, self).__init__(config)
        self.make_model()

    def make_model(self):
        """Makes the vlae model. Inherits from Encoder and Decoder classes

        Returns:
            tf.keras.Model: the compiled vlae model
        """

        def weighted_reconstruction_loss(x, xhat):
            """Weighted negative log likelihood

            Args:
                x (tf.tensor): model input
                xhat (tf.tensor): reconstructed output

            Returns:
                tf.float32: weighted reconstruction loss
            """
            return self.config.model.recon_loss_weight * tf.math.reduce_mean(
                tf.math.square(xhat - x)
            )

        # def weighted_reconstruction_loss(x, xhat):
        #     cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=xhat, labels=x)
        #     logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])

        self.inputs = tf.keras.layers.Input(self.config.model.input_shape)

        # denoise training
        if self.config.model.denoise is True:
            with tf.name_scope("noise_layer"):
                noise_layers = tf.keras.Sequential(
                    [
                        NoiseLayer(),
                        tf.keras.layers.GaussianNoise(self.config.model.noise_ratio),
                    ],
                    name="noise_layer",
                )
                noisy_inputs = noise_layers(self.inputs)

            # encoded noisy samples
            self.h_1, self.h_2, self.h_3, self.h_4 = self.encoder(noisy_inputs)

        else:
            # encoded samples
            self.h_1, self.h_2, self.h_3, self.h_4 = self.encoder(self.inputs)

        # variational layers
        with tf.name_scope("z_1"):
            self.z_1 = VariationalLayer(
                size=self.config.model.latent_size,
                mu_prior=self.config.model.mu_prior,
                sigma_prior=self.config.model.sigma_prior,
                use_kl=self.config.model.use_kl,
                kl_coef=self.config.model.kl_coef,
                use_mmd=self.config.model.use_mmd,
                mmd_coef=self.config.model.mmd_coef,
                sigmas=self.config.model.sigmas,
                name="z_1",
            )(self.h_1)

        with tf.name_scope("z_2"):
            self.z_2 = VariationalLayer(
                size=self.config.model.latent_size,
                mu_prior=self.config.model.mu_prior,
                sigma_prior=self.config.model.sigma_prior,
                use_kl=self.config.model.use_kl,
                kl_coef=self.config.model.kl_coef,
                use_mmd=self.config.model.use_mmd,
                mmd_coef=self.config.model.mmd_coef,
                sigmas=self.config.model.sigmas,
                name="z_2",
            )(self.h_2)

        with tf.name_scope("z_3"):
            self.z_3 = VariationalLayer(
                size=self.config.model.latent_size,
                mu_prior=self.config.model.mu_prior,
                sigma_prior=self.config.model.sigma_prior,
                use_kl=self.config.model.use_kl,
                kl_coef=self.config.model.kl_coef,
                use_mmd=self.config.model.use_mmd,
                mmd_coef=self.config.model.mmd_coef,
                sigmas=self.config.model.sigmas,
                name="z_3",
            )(self.h_3)

        with tf.name_scope("z_4"):
            self.z_4 = VariationalLayer(
                size=self.config.model.latent_size,
                mu_prior=self.config.model.mu_prior,
                sigma_prior=self.config.model.sigma_prior,
                use_kl=self.config.model.use_kl,
                kl_coef=self.config.model.kl_coef,
                use_mmd=self.config.model.use_mmd,
                mmd_coef=self.config.model.mmd_coef,
                sigmas=self.config.model.sigmas,
                name="z_4",
            )(self.h_4)

        self.outputs = self.decoder([self.z_1, self.z_2, self.z_3, self.z_4])

        # perceptual loss layer
        if self.config.model.use_perceptual_loss is True:
            with tf.name_scope("perceptual_loss"):
                (_, self.outputs) = PerceptualLossLayer(
                    perceptual_loss_model=self.config.model.perceptual_loss_model,
                    pereceptual_loss_layers=self.config.model.pereceptual_loss_layers,
                    perceptual_loss_layer_weights=self.config.model.perceptual_loss_layer_weights,
                    model_input_shape=self.config.model.input_shape,
                    name="perceptual_loss",
                )([self.inputs, self.outputs])

        # define and compile the model
        self.model = tf.keras.Model(self.inputs, self.outputs, name="vlae")
        self.model.summary()

        if self.config.model.use_kl:
            # self.model.compile(
            #     tf.keras.optimizers.Adam(),
            #     kl_loss,
            #     metrics=[kl_loss])
            self.model.compile(tf.keras.optimizers.Adam(), "mse")
        else:
            self.model.compile(
                tf.keras.optimizers.Adam(),
                weighted_reconstruction_loss,
                metrics=[weighted_reconstruction_loss],
            )

    def load(self, checkpoint_path):
        """loads model weights from a saved checkpoint

        Args:
            checkpoint_path (str): file location of save model weights

        Raises:
            Exception: must build model prior to loading weights
        """
        if self.model is None:
            raise Exception("You need to build the model first.")

        print("Loading model checkpoint {} ...\n".format(checkpoint_path))
        self.model.load_weights(checkpoint_path, by_name=True)
        print("Model loaded")
