"""
Copyright (c) 2020 R. Ian Etheredge All rights reserved.

This work is licensed under the terms of the MIT license.
For a copy, see <https://opensource.org/licenses/MIT>.
"""
import tensorflow as tf
import tensorflow_probability as tfp

import functools


def gaussian_kernel_matrix(x, y, sigmas):
    beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))
    dist = compute_pairwise_distances(x, y)
    s = tf.matmul(beta, tf.reshape(dist, (1, -1)))
    return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))

def compute_pairwise_distances(x, y): 
    if not len(x.get_shape()) == len(y.get_shape()) == 2:
        raise ValueError('Both inputs should be matrices.')
    if x.get_shape().as_list()[1] != y.get_shape().as_list()[1]:
        raise ValueError('The number of features should be the same.')
    norm = lambda x: tf.reduce_sum(tf.square(x), 1)
    return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))

def maximum_mean_discrepancy(x, y, kernel):
    cost = tf.reduce_mean(kernel(x, x))
    cost += tf.reduce_mean(kernel(y, y))
    cost -= 2 * tf.reduce_mean(kernel(x, y))
    cost = tf.where(cost > 0, cost, 0)
    return cost


class VariationalLayer(tf.keras.layers.Layer):
    def __init__(self, size=2, mu_prior=0., sigma_prior=1.,
                use_kl=False, kl_coef=1.0, use_mmd=True, 
                sigmas=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20,
                25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6],  # all the sigmas!
                mmd_coef=100.0, name=None, **kwargs):
        super().__init__(**kwargs)

        self.mu_layer = tf.keras.layers.Dense(size)
        self.sigma_layer = tf.keras.layers.Dense(size)

        self.use_kl = use_kl
        self.use_mmd = use_mmd

        if use_kl is True:
            self.kl_coef = tf.Variable(kl_coef, trainable=False, name='kl_coef')

        if use_mmd is True:
            self.sigmas = sigmas
            self.kernel_f = functools.partial(
                gaussian_kernel_matrix, sigmas=tf.constant(self.sigmas)
            )
            functools.update_wrapper(self.kernel_f, gaussian_kernel_matrix)

            self.mmd_coef = mmd_coef

        self.mu_prior = tf.constant(mu_prior, dtype=tf.float32, shape=(size,))
        self.sigma_prior = tf.constant(sigma_prior, dtype=tf.float32, shape=(size,))

    def add_mmd_loss(self, z, z_prior):
        mmd = maximum_mean_discrepancy(z, z_prior, kernel=self.kernel_f)
        mmd = tf.maximum(1e-4, mmd) * self.mmd_coef
        self.add_loss(mmd)
        self.add_metric(mmd, 'mean', 'mmd_discrepancy')

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

    def call(self, inputs):
        if self.use_mmd:
            mu = self.mu_layer(inputs)
            log_sigma = self.sigma_layer(inputs)
            sigma_square = tf.exp(log_sigma * 0.5)
            z = mu + (log_sigma * tf.random.normal(shape=tf.shape(sigma_square)))
            z_prior = tfp.distributions.MultivariateNormalDiag(
                self.mu_prior, self.sigma_prior
                ).sample(tf.shape(z)[0])
            self.add_mmd_loss(z, z_prior)

        if self.use_kl:
            mu = self.mu_layer(inputs)
            log_sigma = self.sigma_layer(inputs)
            sigma_square = tf.exp(log_sigma * 0.5)
            z = mu + (log_sigma * tf.random.normal(shape=tf.shape(sigma_square)))
            self.use_kl_divergence(
                mu,
                sigma_square,
                self.mu_prior,
                self.sigma_prior)

        return z

    def get_config(self):
        base_config = super(VariationalLayer, self).get_config()
        config = {
            'use_kl': self.use_kl,
            'use_mmd': self.use_mmd,
            'mmd_coef': self.mmd_coef,
            'kernel_f': self.kernel_f,
            'sigmas': self.sigmas,
        }

        return dict(list(base_config.items()) + list(config.items()))
