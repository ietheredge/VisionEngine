from base.base_model import BaseModel

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


class NormalVariational(tf.keras.layers.Layer):
    def __init__(self, size=2, mu_prior=0., sigma_prior=1.,
                 use_kl=False, kl_coef=1.0,
                 use_mmd=True, mmd_coef=100.0, name=None, **kwargs):
        super().__init__(**kwargs)
        self.mu_layer = tf.keras.layers.Dense(size)
        self.sigma_layer = tf.keras.layers.Dense(size)
        self.mu_prior = tf.constant(mu_prior, dtype=tf.float32, shape=(size,))
        self.sigma_prior = tf.constant(
            sigma_prior, dtype=tf.float32, shape=(size,)
            )

        self.use_kl = use_kl
        self.kl_coef = tf.Variable(kl_coef, trainable=False, name='kl_coef')
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
        kl = tf.reduce_mean(
            self.kl_coef * tf.reduce_sum(
                tf.math.log(p_sigma) -
                tf.math.log(q_sigma) -
                .5 * (1. - (q_sigma**2 + r**2) / p_sigma**2), axis=1)
                )

        self.add_loss(kl)
        self.add_metric(kl, 'mean', 'kl_divergence')

    def use_mm_discrepancy(self, z, z_prior):
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
            sigma_square = tf.exp(log_sigma)
            z = mu + sigma_square * tf.random.normal(tf.shape(sigma_square))
            z_prior = tfp.distributions.MultivariateNormalDiag(
                self.mu_prior, self.sigma_prior
                ).sample(tf.shape(z)[0])
            self.add_mm_discrepancy(z, z_prior)

        if self.use_kl:
            mu = self.mu_layer(inputs)
            log_sigma = self.sigma_layer(inputs)
            sigma_square = tf.exp(log_sigma)
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
            'kl_coef': self.kl_coef,
            'use_mmd': self.use_mmd,
            'mmd_coef': self.mmd_coef,
            'kernel_f': self.kernel_f,
            'mu_prior': self.mu_prior,
            'mu_layer': self.mu_layer,
            'sigma_prior': self.sigma_prior,
            'sigma_layer': self.sigma_layer
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
            noised(), inputs, training=training)

    def get_config(self):
        config = {'ratio': self.ratio,
                  'masking': self.masking}
        base_config = super(SaltAndPepper, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


class Encoder(BaseModel):
    def __init__(self, config):
        super(Encoder, self).__init__(config)
        self.build_encoder()

    def make_encoder(self, latent_size):
        self.inputs = tf.keras.layers.Input(
            shape=self.config.model.input_shape, name='input')

        with tf.name_scope('noise_layer'):
            noise_layers = tf.keras.Sequential([
                SaltAndPepper(),
                tf.keras.layers.GaussianNoise(self.config.model.noise_ratio)
                ], name='noise_layer')

            noisy_inputs = noise_layers(self.inputs)

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

            h_1 = h_1_layers(noisy_inputs)
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

        with tf.name_scope('encoder'):
            self.encoder = tf.keras.Model(
                self.inputs, self.encoder_outputs, name='encoder')


class Decoder(BaseModel):
    def __init__(self, config):
        super(Decoder, self).__init__(config)
        self.build_encoder()

    def make_decoder(self):
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
                tf.keras.layers.Reshape((16, 16, 512))], name='z_tilde_4')

            z_tilde_4 = z_tilde_4_layers(z_4)

        with tf.name_scope('z_tilde_3'):
            z_3 = tf.keras.layers.Dense(
                16*16*512, kernel_regularizer=tf.keras.regularizers.l2(
                    self.config.model.kernel_l2_regularize))(self.z_3_input)

            z_3 = tf.keras.layers.BatchNormalization()(z_3)
            z_3 = tf.keras.layers.ReLU()(z_3)
            z_3 = tf.keras.layers.Reshape((16, 16, 512))(z_3)
            z_tilde_3_layers = tf.keras.Sequential([
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
                tf.keras.layers.ReLU()], name='z_tilde_3')

            input_z_tilde_3 = tf.keras.layers.Concatenate()([z_tilde_4, z_3])
            z_tilde_3 = z_tilde_3_layers(input_z_tilde_3)

        with tf.name_scope('z_tilde_2'):
            z_2 = tf.keras.layers.Dense(
                32*32*256, kernel_regularizer=tf.keras.regularizers.l2(
                    2.5e-5))(self.z_2_input)
            z_2 = tf.keras.layers.BatchNormalization()(z_2)
            z_2 = tf.keras.layers.ReLU()(z_2)
            z_2 = tf.keras.layers.Reshape((32, 32, 256))(z_2)
            z_tilde_2_layers = tf.keras.Sequential([
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
                tf.keras.layers.ReLU()], name='z_tilde_2')

            input_z_tilde_2 = tf.keras.layers.Concatenate()([z_tilde_3, z_2])
            z_tilde_2 = z_tilde_2_layers(input_z_tilde_2)

        with tf.name_scope('z_tilde_1'):
            z_1 = tf.keras.layers.Dense(
                128*128*64, kernel_regularizer=tf.keras.regularizers.l2(
                    self.config.model.kernel_l2_regularize))(self.z_1_input)

            z_1 = tf.keras.layers.BatchNormalization()(z_1)
            z_1 = tf.keras.layers.ReLU()(z_1)
            z_1 = tf.keras.layers.Reshape((128, 128, 64))(z_1)
            z_tilde_1_layers = tf.keras.layers.Sequential([
                tf.keras.layers.Conv2DTranspose(
                    128, 4, 2, padding='same',
                    kernel_regularizer=tf.keras.regularizers.l2(
                        self.config.model.kernel_l2_regularize)),

                tf.keras.layers.Conv2DTranspose(
                    3, 4, 1, padding='same',
                    kernel_regularizer=tf.keras.regularizers.l2(
                        self.config.model.kernel_l2_regularize)),

                tf.keras.layers.Activation('sigmoid')], name='z_tilde_1')

            input_z_tilde_1 = tf.keras.layers.Concatenate()([z_tilde_2, z_1])
            self.decoder_outputs = z_tilde_1_layers(input_z_tilde_1)
            self.decoder_inputs = [
                self.z_1_input, self.z_2_input, self.z_3_input, self.z_4_input
                ]
        with tf.name_scope('decoder'):
            self.decoder = tf.keras.Model(
                self.decoder_inputs,
                self.decoder_outputs,
                name='decoder')


class VAEModel(Decoder, Encoder):
    def __init__(self, config):
        super(VAEModel, self).__init__(config)
        self.build_model()

    def build_model(self):

        def custom_loss(x, xhat):
            return self.config.model.recon_loss_weight * \
                tf.losses.mean_squared_error(
                    tf.keras.layers.Flatten()(x),
                    tf.keras.layers.Flatten()(xhat)) \
                * np.prod(self.config.input_shape)

        inputs = tf.keras.layers.Input(self.config.model.input_shape)
        h_1, h_2, h_3, h_4 = self.encoder(inputs)
        z_1 = NormalVariational(
            size=self.config.model.latent_size,
            mu_prior=self.config.model.mu_prior,
            sigma_prior=self.config.model.sigma_prior,
            use_kl=self.config.model.use_kl,
            kl_coef=self.config.model.kl_coef,
            use_mmd=self.config.model.use_mmd,
            mmd_coef=self.config.model.mmd_coef,
            name='z_1_latent')(h_1)

        z_2 = NormalVariational(
            size=self.config.model.latent_size,
            mu_prior=self.config.model.mu_prior,
            sigma_prior=self.config.model.sigma_prior,
            use_kl=self.config.model.use_kl,
            kl_coef=self.config.model.kl_coef,
            use_mmd=self.config.model.use_mmd,
            mmd_coef=self.config.model.mmd_coef,
            name='z_2_latent')(h_2)

        z_3 = NormalVariational(
            size=self.config.model.latent_size,
            mu_prior=self.config.model.mu_prior,
            sigma_prior=self.config.model.sigma_prior,
            use_kl=self.config.model.use_kl,
            kl_coef=self.config.model.kl_coef,
            use_mmd=self.config.model.use_mmd,
            mmd_coef=self.config.model.mmd_coef,
            name='z_3_latent')(h_3)

        z_4 = NormalVariational(
            size=self.config.model.latent_size,
            mu_prior=self.config.model.mu_prior,
            sigma_prior=self.config.model.sigma_prior,
            use_kl=self.config.model.use_kl,
            kl_coef=self.config.model.kl_coef,
            use_mmd=self.config.model.use_mmd,
            mmd_coef=self.config.model.mmd_coef,
            name='z_4_latent')(h_4)

        self.decoded = self.decoder([z_1, z_2, z_3, z_4])
        vlae = tf.keras.Model(self.inputs, self.decoded, name='vlae')
        vlae.summary()
        vlae.compile(tf.keras.optimizers.Adam(), custom_loss)
        return vlae

    def load(self, checkpoint_path):
        if self.model is None:
            raise Exception("You need to build the model first.")

        print("Loading model checkpoint {} ...\n".format(checkpoint_path))
        self.model.load_model(
            checkpoint_path,
            custom_objects={'NormalVariational':
                            NormalVariational,
                            'SaltAndPepper':
                            SaltAndPepper,
                            }
            )

        print("Model loaded")
