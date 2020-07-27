"""
Copyright (c) 2020 R. Ian Etheredge All rights reserved.

This work is licensed under the terms of the MIT license.
For a copy, see <https://opensource.org/licenses/MIT>.
"""
from VisionEngine.base.base_model import BaseModel

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

import VisionEngine.models.wppvae as wppvae


class SqueezeExcite(tf.keras.layers.Layer):
    def __init__(self, c, r=16, **kwargs):
        super(SqueezeExcite, self).__init__(**kwargs)
        self.c = c
        self.r = r

    def build(self, input_shape):
        self.se = tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(self.c // self.r, use_bias=False),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(self.c, use_bias=False),
            tf.keras.layers.Activation('sigmoid')])

    def call(self, layer_inputs, **kwargs):
        return self.se(layer_inputs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'c': self.c,
            'r': self.r
        }
        base_config = \
            super(SqueezeExcite, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SpectralNormalization(tf.keras.layers.Wrapper):
    def __init__(self, layer, iteration=1, eps=1e-12, training=True, **kwargs):
        self.iteration = iteration
        self.eps = eps
        self.do_power_iteration = training
        if not isinstance(layer, tf.keras.layers.Layer):
            raise ValueError(
                'Please initialize `TimeDistributed` layer with a '
                '`Layer` instance. You passed: {input}'.format(input=layer))
        super(SpectralNormalization, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        self.layer.build(input_shape)

        self.w = self.layer.kernel
        self.w_shape = self.w.shape.as_list()

        self.v = self.add_weight(shape=(1, self.w_shape[0] * self.w_shape[1] * self.w_shape[2]),
                                 initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                                 trainable=False,
                                 name='sn_v',
                                 dtype=tf.float32)

        self.u = self.add_weight(shape=(1, self.w_shape[-1]),
                                 initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                                 trainable=False,
                                 name='sn_u',
                                 dtype=tf.float32)

        super(SpectralNormalization, self).build()

    def call(self, inputs):
        self.update_weights()
        output = self.layer(inputs)
        self.restore_weights()  # Restore weights because of this formula "W = W - alpha * W_SN`"
        return output
    
    def update_weights(self):
        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])
        
        u_hat = self.u
        v_hat = self.v  # init v vector

        if self.do_power_iteration:
            for _ in range(self.iteration):
                v_ = tf.matmul(u_hat, tf.transpose(w_reshaped))
                v_hat = v_ / (tf.reduce_sum(v_**2)**0.5 + self.eps)

                u_ = tf.matmul(v_hat, w_reshaped)
                u_hat = u_ / (tf.reduce_sum(u_**2)**0.5 + self.eps)

        sigma = tf.matmul(tf.matmul(v_hat, w_reshaped), tf.transpose(u_hat))
        self.u.assign(u_hat)
        self.v.assign(v_hat)

        self.layer.kernel.assign(self.w / sigma)

    def restore_weights(self):
        self.layer.kernel.assign(self.w)


class PerceptualLossLayer(tf.keras.layers.Layer):
    def __init__(self, perceptual_loss_model,
                 pereceptual_loss_layers, perceptual_loss_layer_weights,
                 model_input_shape, name, **kwargs):
        super(PerceptualLossLayer, self).__init__(**kwargs)
        self.loss_model_type = perceptual_loss_model
        self.layers = pereceptual_loss_layers
        self.layer_weights = perceptual_loss_layer_weights
        self.model_input_shape = [256, 256, 3]

    def build(self, input_shape):
        if self.loss_model_type == 'vgg':
            self.loss_model_ = tf.keras.applications.VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=self.model_input_shape
                )
            self.loss_model_.trainable = False

            for layer in self.loss_model_.layers:
                layer.trainable = False

            self.loss_layers = [
                tf.keras.layers.BatchNormalization()(self.loss_model_.layers[i].output)
                for i in self.layers
                ]

            self.loss_model = tf.keras.Model(
                self.loss_model_.inputs,
                self.loss_layers,
                )
        else:
            raise NotImplementedError
        super(PerceptualLossLayer, self).build(input_shape)

    def call(self, layer_inputs, **kwargs):
        y_true = layer_inputs[0]
        y_pred = layer_inputs[1]

        self.sample_ = self.loss_model(y_true)
        self.reconstruction_ = self.loss_model(y_pred)

        self.perceptual_loss = 0.
        for i in range(len(self.reconstruction_)):
            shape = tf.cast(tf.shape(self.reconstruction_[i]), dtype='float32')
            self.perceptual_loss += tf.math.reduce_mean(
                self.layer_weights[i] *
                (tf.math.reduce_sum(tf.math.square(self.sample_[i] - self.reconstruction_[i])) /
                    (shape[-1] * shape[1] * shape[1]))
            )

        perceptual_loss = tf.cast(self.perceptual_loss, dtype='float32')
        self.add_loss(perceptual_loss)
        self.add_metric(perceptual_loss, 'mean', 'perceptual_loss')

        return [y_true, y_pred]

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'perceptual_loss_model': self.loss_model_type,
            'pereceptual_loss_layers': self.layers,
            'perceptual_loss_layer_weights':
                self.layer_weights,
            'model_input_shape': self.model_input_shape,
        }
        base_config = \
            super(PerceptualLossLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class NormalVariational(tf.keras.layers.Layer):
    def __init__(self, size=2, mu_prior=0., sigma_prior=1.,
                    use_kl=False, kl_coef=1.0,
                    use_mmd=True, mmd_coef=100.0, name=None, **kwargs):
        super().__init__(**kwargs)
        self.mu_layer = tf.keras.layers.Dense(size)
        self.sigma_layer = tf.keras.layers.Dense(size)
        if use_kl is True:
            # self.sigma_layer = tf.keras.layers.Dense(size)
            self.kl_coef = tf.Variable(kl_coef, trainable=False, name='kl_coef')
        self.mu_prior = tf.constant(mu_prior, dtype=tf.float32, shape=(size,))
        self.sigma_prior = tf.constant(
            sigma_prior, dtype=tf.float32, shape=(size,)
            )

        self.use_kl = use_kl
        self.use_mmd = use_mmd
        self.mmd_coef = mmd_coef
        self.kernel_f = self._rbf

    def _rbf(self, x, y):
        x_size = tf.shape(x)[0]
        y_size = tf.shape(y)[0]
        dim = tf.shape(x)[1]
        tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])),
                          tf.stack([1, y_size, 1]))

        tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])),
                          tf.stack([x_size, 1, 1]))

        return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) /
                      tf.cast(dim, tf.float32))

    def use_kl_divergence(self, q_mu, q_sigma, p_mu, p_sigma):
        r = q_mu - p_mu
        kl = self.kl_coef * tf.reduce_mean(
            tf.reduce_sum(
                tf.math.log(p_sigma) -
                tf.math.log(q_sigma) -
                .5 * (1. - (q_sigma**2 + r**2) / p_sigma**2), axis=1
                )
            )
        self.add_loss(kl)
        self.add_metric(kl, 'mean', 'kl_divergence')

    def add_mm_discrepancy(self, z, z_prior):
        k_prior = self.kernel_f(z_prior, z_prior)
        k_post = self.kernel_f(z, z)
        k_prior_post = self.kernel_f(z_prior, z)
        mmd = tf.reduce_mean(k_prior) + \
            tf.reduce_mean(k_post) - \
            2 * tf.reduce_mean(k_prior_post)

        mmd = tf.multiply(self.mmd_coef,  mmd, name='mmd')
        self.add_loss(mmd)
        self.add_metric(mmd, 'mean', 'mmd_discrepancy')

    def call(self, inputs):
        if self.use_mmd:
            mu = self.mu_layer(inputs)
            log_sigma = self.sigma_layer(inputs)
            sigma_square = tf.exp(log_sigma * 0.5)
            z = mu + (log_sigma * tf.random.normal(shape=tf.shape(sigma_square)))
            z_prior = tfp.distributions.MultivariateNormalDiag(
                self.mu_prior, self.sigma_prior
                ).sample(tf.shape(z)[0])
            self.add_mm_discrepancy(z, z_prior)

        if self.use_kl:
            mu = self.mu_layer(inputs)
            log_sigma = self.sigma_layer(inputs)
            sigma_square = tf.exp(log_sigma * 0.5)
            self.use_kl_divergence(
                mu,
                sigma_square,
                self.mu_prior,
                self.sigma_prior)

        return z

    def get_config(self):
        base_config = super(NormalVariational, self).get_config()
        config = {
            'use_kl': self.use_kl,
            'use_mmd': self.use_mmd,
            'mmd_coef': self.mmd_coef,
            'kernel_f': self.kernel_f,
        }

        return dict(list(base_config.items()) + list(config.items()))


class SaltAndPepper(tf.keras.layers.Layer):
    def __init__(self, ratio=0.9, **kwargs):
        super(SaltAndPepper, self).__init__(**kwargs)
        self.masking = True
        self.ratio = ratio

    def call(self, inputs, training=None):
        def noised():
            shp = tf.keras.backend.shape(inputs)[1:]
            mask_select = tf.keras.backend.random_binomial(
                shape=shp, p=self.ratio)

            # salt and pepper have the same chance
            mask_noise = tf.keras.backend.random_binomial(shape=shp, p=0.1)
            out = (inputs * (mask_select)) + mask_noise
            return out

        return tf.keras.backend.in_train_phase(
            noised, inputs, training=training)

    def get_config(self):
        config = {'ratio': self.ratio,
                  'masking': self.masking}
        base_config = super(SaltAndPepper, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


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
                        SaltAndPepper(),
                        tf.keras.layers.GaussianNoise(self.config.model.noise_ratio)
                        ], name='noise_layer')

                    noisy_inputs = noise_layers(self.encoder_inputs)

            with tf.name_scope('z_1'):
                h_1_layers = tf.keras.Sequential([
                    tf.keras.layers.Input(self.config.model.input_shape),
                    SpectralNormalization(tf.keras.layers.Conv2D(
                        8, 3, padding='same')),
                    tf.keras.layers.BatchNormalization(),
                    SpectralNormalization(tf.keras.layers.Conv2D(64, 3, padding='same')),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Activation(self.config.model.encoder_activations),
                    SpectralNormalization(tf.keras.layers.Conv2D(
                        16, 3, padding='same')),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Activation(self.config.model.encoder_activations),
                    SpectralNormalization(tf.keras.layers.Conv2D(
                        16, 3, padding='same')),
                    tf.keras.layers.BatchNormalization(),
                    SpectralNormalization(tf.keras.layers.Conv2D(64, 3, padding='same')),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Activation(self.config.model.encoder_activations),
                    SpectralNormalization(tf.keras.layers.Conv2D(
                        32, 3, padding='same')),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Activation(self.config.model.encoder_activations)], name='h_1')
                if self.config.modeul.denoise is True:
                    h_1 = h_1_layers(noisy_inputs)
                else:
                    h_1 = h_1_layers(self.encoder_inputs)

                h_1_flatten = SqueezeExcite(c=32)(h_1)

            with tf.name_scope('z_2'):
                h_2_layers = tf.keras.Sequential([
                    SpectralNormalization(tf.keras.layers.Conv2D(
                        32, 3, padding='same')),
                    tf.keras.layers.BatchNormalization(),
                    SpectralNormalization(tf.keras.layers.Conv2D(64, 3, padding='same')),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Activation(self.config.model.encoder_activations),
                    SpectralNormalization(tf.keras.layers.Conv2D(
                        64, 3, padding='same')),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Activation(self.config.model.encoder_activations),
                    SpectralNormalization(tf.keras.layers.Conv2D(
                        64, 3, padding='same')),
                    tf.keras.layers.BatchNormalization(),
                    SpectralNormalization(tf.keras.layers.Conv2D(64, 3, padding='same')),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Activation(self.config.model.encoder_activations),
                    SpectralNormalization(tf.keras.layers.Conv2D(
                        128, 3, padding='same')),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Activation(self.config.model.encoder_activations)], name='h_2')

                h_2 = h_2_layers(h_1)
                h_2_flatten = SqueezeExcite(c=128)(h_2)

            with tf.name_scope('z_3'):
                h_3_layers = tf.keras.Sequential([
                    SpectralNormalization(tf.keras.layers.Conv2D(
                        128, 3, padding='same')),
                    tf.keras.layers.BatchNormalization(),
                    SpectralNormalization(tf.keras.layers.Conv2D(64, 3, padding='same')),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Activation(self.config.model.encoder_activations),
                    SpectralNormalization(tf.keras.layers.Conv2D(
                        256, 3, padding='same')),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Activation(self.config.model.encoder_activations),
                    SpectralNormalization(tf.keras.layers.Conv2D(
                        256, 3, padding='same')),
                    tf.keras.layers.BatchNormalization(),
                    SpectralNormalization(tf.keras.layers.Conv2D(64, 3, padding='same')),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Activation(self.config.model.encoder_activations),
                    SpectralNormalization(tf.keras.layers.Conv2D(
                        512, 3, padding='same')),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Activation(self.config.model.encoder_activations)], name='h_3')

                h_3 = h_3_layers(h_2)
                h_3_flatten = SqueezeExcite(c=512)(h_3)

            with tf.name_scope('z_4'):
                h_4_layers = tf.keras.Sequential([
                    SpectralNormalization(tf.keras.layers.Conv2D(
                        512, 3, padding='same')),
                    tf.keras.layers.BatchNormalization(),
                    SpectralNormalization(tf.keras.layers.Conv2D(64, 3, padding='same')),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Activation(self.config.model.encoder_activations),
                    SpectralNormalization(tf.keras.layers.Conv2D(
                        1024, 3, padding='same')),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Activation(self.config.model.encoder_activations),
                    SpectralNormalization(tf.keras.layers.Conv2D(
                        1024, 3, padding='same')),
                    tf.keras.layers.BatchNormalization(),
                    SpectralNormalization(tf.keras.layers.Conv2D(64, 3, padding='same')),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Activation(self.config.model.encoder_activations),
                    SpectralNormalization(tf.keras.layers.Conv2D(
                        2048, 3, padding='same')),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Activation(self.config.model.encoder_activations)], name='h_4')

                h_4 = h_4_layers(h_3)
                h_4_flatten = SqueezeExcite(c=2048)(h_4)

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
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Activation(self.config.model.decoder_activations),
                    tf.keras.layers.Dense(
                        16*16*512, kernel_regularizer=tf.keras.regularizers.l2(
                            self.config.model.kernel_l2_regularize)),

                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Activation(self.config.model.decoder_activations),
                    tf.keras.layers.Reshape((16, 16, 512))], name='z_tilde_4')

                z_tilde_4 = z_tilde_4_layers(z_4)

            with tf.name_scope('z_tilde_3'):
                z_3 = tf.keras.layers.Dense(
                    16*16*512)(self.z_3_input)

                z_3 = tf.keras.layers.BatchNormalization()(z_3)
                z_3 = tf.keras.layers.Activation(self.config.model.decoder_activations)(z_3)
                z_3 = tf.keras.layers.Reshape((16, 16, 512))(z_3)
                z_tilde_3_layers = tf.keras.Sequential([
                    SpectralNormalization(tf.keras.layers.Conv2DTranspose(
                        512, 4, 1, padding='same')),

                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Activation(self.config.model.decoder_activations),
                    SpectralNormalization(tf.keras.layers.Conv2DTranspose(
                        256, 4, 2, padding='same')),

                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Activation(self.config.model.decoder_activations),
                    SpectralNormalization(tf.keras.layers.Conv2DTranspose(
                        256, 4, 1, padding='same')),

                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Activation(self.config.model.decoder_activations)], name='z_tilde_3')

                input_z_tilde_3 = tf.keras.layers.Concatenate()([z_tilde_4, z_3])
                z_tilde_3 = z_tilde_3_layers(input_z_tilde_3)

            with tf.name_scope('z_tilde_2'):
                z_2 = tf.keras.layers.Dense(
                    32*32*256)(self.z_2_input)
                z_2 = tf.keras.layers.BatchNormalization()(z_2)
                z_2 = tf.keras.layers.Activation(self.config.model.decoder_activations)(z_2)
                z_2 = tf.keras.layers.Reshape((32, 32, 256))(z_2)
                
                z_tilde_2_layers = tf.keras.Sequential([
                    SpectralNormalization(tf.keras.layers.Conv2DTranspose(
                        128, 4, 2, padding='same')),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Activation(self.config.model.decoder_activations),
                    SpectralNormalization(tf.keras.layers.Conv2DTranspose(
                        128, 4, 1, padding='same')),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Activation(self.config.model.decoder_activations),
                    SpectralNormalization(tf.keras.layers.Conv2DTranspose(
                        64, 4, 2, padding='same')),

                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Activation(self.config.model.decoder_activations)], name='z_tilde_2')

                input_z_tilde_2 = tf.keras.layers.Concatenate()([z_tilde_3, z_2])
                z_tilde_2 = z_tilde_2_layers(input_z_tilde_2)

            with tf.name_scope('z_tilde_1'):
                z_1 = tf.keras.layers.Dense(128*128*64)(self.z_1_input)

                z_1 = tf.keras.layers.BatchNormalization()(z_1)
                z_1 = tf.keras.layers.Activation(self.config.model.decoder_activations)(z_1)
                z_1 = tf.keras.layers.Reshape((128, 128, 64))(z_1)
                z_tilde_1_layers = tf.keras.Sequential([
                    SpectralNormalization(tf.keras.layers.Conv2DTranspose(
                        128, 4, 2, padding='same')),

                    SpectralNormalization(tf.keras.layers.Conv2DTranspose(
                        self.config.model.input_shape[2], 4, 1, padding='same')),

                    tf.keras.layers.Activation(self.config.model.last_activation)
                    ], name='z_tilde_1')

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
        self.inputs = tf.keras.layers.Input(self.config.model.input_shape)
        self.h_1, self.h_2, self.h_3, self.h_4 = self.encoder(self.inputs)

        if self.config.model.use_wppvae is True:
            self.h_1 = wppvae.DenseSiren(
                self.config.model.wppvae_size,
                activation=self.config.model.siren_activation,
                w0=self.config.model.wppvae_w0,
                bias_initializer=self.config.model.siren_bias_init,
                kernel_initializer=self.config.model.siren_init_kernel_init)(self.h_1)

            self.h_1 = wppvae.DenseSiren(
                self.config.model.wppvae_size,
                activation=self.config.model.siren_activation,
                w0=self.config.model.wppvae_w0,
                kernel_initializer=self.config.model.siren_kernel_init,
                bias_initializer=self.config.model.siren_bias_init,
                kernel_regularizer=self.config.model.siren_kernel_regularizer,
                kernel_constraint=self.config.model.siren_kernel_constraint,
                activity_regularizer=self.config.model.siren_activity_regularizer)(self.h_1)

            self.h_2 = wppvae.DenseSiren(
                self.config.model.wppvae_size,
                activation=self.config.model.siren_activation,
                w0=self.config.model.wppvae_w0,
                bias_initializer=self.config.model.siren_bias_init,
                kernel_initializer=self.config.model.siren_init_kernel_init)(self.h_2)

            self.h_2 = wppvae.DenseSiren(
                self.config.model.wppvae_size,
                activation=self.config.model.siren_activation,
                w0=self.config.model.wppvae_w0,
                kernel_initializer=self.config.model.siren_kernel_init,
                bias_initializer=self.config.model.siren_bias_init,
                kernel_regularizer=self.config.model.siren_kernel_regularizer,
                kernel_constraint=self.config.model.siren_kernel_constraint,
                activity_regularizer=self.config.model.siren_activity_regularizer)(self.h_2)

            self.h_3 = wppvae.DenseSiren(
                self.config.model.wppvae_size,
                activation=self.config.model.siren_activation,
                w0=self.config.model.wppvae_w0,
                bias_initializer=self.config.model.siren_bias_init,
                kernel_initializer=self.config.model.siren_init_kernel_init)(self.h_3)

            self.h_3 = wppvae.DenseSiren(
                self.config.model.wppvae_size,
                activation=self.config.model.siren_activation,
                w0=self.config.model.wppvae_w0,
                kernel_initializer=self.config.model.siren_kernel_init,
                bias_initializer=self.config.model.siren_bias_init,
                kernel_regularizer=self.config.model.siren_kernel_regularizer,
                kernel_constraint=self.config.model.siren_kernel_constraint,
                activity_regularizer=self.config.model.siren_activity_regularizer)(self.h_3)

            self.h_4 = wppvae.DenseSiren(
                self.config.model.wppvae_size,
                activation=self.config.model.siren_activation,
                w0=self.config.model.wppvae_w0,
                bias_initializer=self.config.model.siren_bias_init,
                kernel_initializer=self.config.model.siren_init_kernel_init)(self.h_4)

            self.h_4 = wppvae.DenseSiren(
                self.config.model.wppvae_size,
                activation=self.config.model.siren_activation,
                w0=self.config.model.wppvae_w0,
                kernel_initializer=self.config.model.siren_kernel_init,
                bias_initializer=self.config.model.siren_bias_init,
                kernel_regularizer=self.config.model.siren_kernel_regularizer,
                kernel_constraint=self.config.model.siren_kernel_constraint,
                activity_regularizer=self.config.model.siren_activity_regularizer)(self.h_4)

        self.z_1 = NormalVariational(
            size=self.config.model.latent_size,
            mu_prior=self.config.model.mu_prior,
            sigma_prior=self.config.model.sigma_prior,
            use_kl=self.config.model.use_kl,
            kl_coef=self.config.model.kl_coef,
            use_mmd=self.config.model.use_mmd,
            mmd_coef=self.config.model.mmd_coef,
            name='z_1_latent')(self.h_1)

        self.z_2 = NormalVariational(
            size=self.config.model.latent_size,
            mu_prior=self.config.model.mu_prior,
            sigma_prior=self.config.model.sigma_prior,
            use_kl=self.config.model.use_kl,
            kl_coef=self.config.model.kl_coef,
            use_mmd=self.config.model.use_mmd,
            mmd_coef=self.config.model.mmd_coef,
            name='z_2_latent')(self.h_2)

        self.z_3 = NormalVariational(
            size=self.config.model.latent_size,
            mu_prior=self.config.model.mu_prior,
            sigma_prior=self.config.model.sigma_prior,
            use_kl=self.config.model.use_kl,
            kl_coef=self.config.model.kl_coef,
            use_mmd=self.config.model.use_mmd,
            mmd_coef=self.config.model.mmd_coef,
            name='z_3_latent')(self.h_3)

        self.z_4 = NormalVariational(
            size=self.config.model.latent_size,
            mu_prior=self.config.model.mu_prior,
            sigma_prior=self.config.model.sigma_prior,
            use_kl=self.config.model.use_kl,
            kl_coef=self.config.model.kl_coef,
            use_mmd=self.config.model.use_mmd,
            mmd_coef=self.config.model.mmd_coef,
            name='z_4_latent')(self.h_4)

        if self.config.model.use_wppvae is True:
            self.z_1 = wppvae.DenseSiren(
                self.config.model.wppvae_size,
                activation=self.config.model.siren_activation,
                w0=self.config.model.wppvae_w0,
                bias_initializer=self.config.model.siren_bias_init,
                kernel_initializer=self.config.model.siren_init_kernel_init)(self.z_1)

            self.z_1 = wppvae.DenseSiren(
                self.config.model.wppvae_size,
                activation=self.config.model.siren_activation,
                w0=self.config.model.wppvae_w0,
                kernel_initializer=self.config.model.siren_kernel_init,
                bias_initializer=self.config.model.siren_bias_init,
                kernel_regularizer=self.config.model.siren_kernel_regularizer,
                kernel_constraint=self.config.model.siren_kernel_constraint,
                activity_regularizer=self.config.model.siren_activity_regularizer)(self.z_1)

            self.z_2 = wppvae.DenseSiren(
                self.config.model.wppvae_size,
                activation=self.config.model.siren_activation,
                w0=self.config.model.wppvae_w0,
                bias_initializer=self.config.model.siren_bias_init,
                kernel_initializer=self.config.model.siren_init_kernel_init)(self.z_2)

            self.z_2 = wppvae.DenseSiren(
                self.config.model.wppvae_size,
                activation=self.config.model.siren_activation,
                w0=self.config.model.wppvae_w0,
                kernel_initializer=self.config.model.siren_kernel_init,
                bias_initializer=self.config.model.siren_bias_init,
                kernel_regularizer=self.config.model.siren_kernel_regularizer,
                kernel_constraint=self.config.model.siren_kernel_constraint,
                activity_regularizer=self.config.model.siren_activity_regularizer)(self.z_2)

            self.z_3 = wppvae.DenseSiren(
                self.config.model.wppvae_size,
                activation=self.config.model.siren_activation,
                w0=self.config.model.wppvae_w0,
                bias_initializer=self.config.model.siren_bias_init,
                kernel_initializer=self.config.model.siren_init_kernel_init)(self.z_3)

            self.z_3 = wppvae.DenseSiren(
                self.config.model.wppvae_size,
                activation=self.config.model.siren_activation,
                w0=self.config.model.wppvae_w0,
                kernel_initializer=self.config.model.siren_kernel_init,
                bias_initializer=self.config.model.siren_bias_init,
                kernel_regularizer=self.config.model.siren_kernel_regularizer,
                kernel_constraint=self.config.model.siren_kernel_constraint,
                activity_regularizer=self.config.model.siren_activity_regularizer)(self.z_3)

            self.z_4 = wppvae.DenseSiren(
                self.config.model.wppvae_size,
                activation=self.config.model.siren_activation,
                w0=self.config.model.wppvae_w0,
                bias_initializer=self.config.model.siren_bias_init,
                kernel_initializer=self.config.model.siren_init_kernel_init)(self.z_4)

            self.z_4 = wppvae.DenseSiren(
                self.config.model.wppvae_size,
                activation=self.config.model.siren_activation,
                w0=self.config.model.wppvae_w0,
                kernel_initializer=self.config.model.siren_kernel_init,
                bias_initializer=self.config.model.siren_bias_init,
                kernel_regularizer=self.config.model.siren_kernel_regularizer,
                kernel_constraint=self.config.model.siren_kernel_constraint,
                activity_regularizer=self.config.model.siren_activity_regularizer)(self.z_4)

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
        self.model.compile(tf.keras.optimizers.Adam(), self.config.model.loss)

    def load(self, checkpoint_path):
        if self.model is None:
            raise Exception("You need to build the model first.")

        print("Loading model checkpoint {} ...\n".format(checkpoint_path))
        self.model.load_weights(checkpoint_path)
        print("Model loaded")


tf.keras.utils.get_custom_objects().update({
    'lrelu': tf.keras.layers.LeakyReLU
})