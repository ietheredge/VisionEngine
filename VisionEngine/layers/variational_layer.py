"""
Copyright (c) 2020 R. Ian Etheredge All rights reserved.

This work is licensed under the terms of the MIT license.
For a copy, see <https://opensource.org/licenses/MIT>.
"""
import tensorflow as tf
import numpy as np

import functools


def gaussian_kernel_matrix(x, y, sigmas):
    beta = 1.0 / (2.0 * (tf.expand_dims(sigmas, 1)))
    dist = compute_pairwise_distances(x, y)
    s = tf.matmul(beta, tf.reshape(dist, (1, -1)))
    return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))


def compute_pairwise_distances(x, y):
    if not len(x.get_shape()) == len(y.get_shape()) == 2:
        raise ValueError("Both inputs should be matrices.")
    if x.get_shape().as_list()[1] != y.get_shape().as_list()[1]:
        raise ValueError("The number of features should be the same.")
    norm = lambda x: tf.reduce_sum(tf.square(x), 1)
    return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))


def maximum_mean_discrepancy(x, y, kernel):
    cost = tf.reduce_mean(kernel(x, x))
    cost += tf.reduce_mean(kernel(y, y))
    cost -= 2 * tf.reduce_mean(kernel(x, y))
    cost = tf.where(cost > 0, cost, 0)
    return cost


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2.0 * np.pi)
    return tf.math.reduce_sum(
        -0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + log2pi), axis=raxis
    )


def kl_divergence(z, mu, sigma, mu_prior, sigma_prior):
    logpz = log_normal_pdf(z, mu_prior, sigma_prior)
    logqz_x = log_normal_pdf(z, mu, sigma)
    kl = -tf.math.reduce_mean(logpz - logqz_x)
    return kl


class VariationalLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        size=2,
        mu_prior=0.0,
        sigma_prior=0.0,
        use_kl=False,
        kl_coef=1.0,
        use_mmd=True,
        sigmas=[
            1e-6,
            1e-5,
            1e-4,
            1e-3,
            1e-2,
            1e-1,
            1,
            5,
            10,
            15,
            20,
            25,
            30,
            35,
            100,
            1e3,
            1e4,
            1e5,
            1e6,
        ],  # all the sigmas!
        mmd_coef=100.0,
        name=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.use_kl = use_kl
        self.use_mmd = use_mmd
        self.mmd_coef = mmd_coef

        if use_kl is True:
            self.mu_layer = tf.keras.layers.Dense(size)
            self.sigma_layer = tf.keras.layers.Dense(size)
            self.kl_coef = tf.Variable(kl_coef, trainable=False, name="kl_coef")

        if use_mmd is True:
            self.z = tf.keras.layers.Dense(size)
            self.sigmas = sigmas
            self.kernel_f = functools.partial(
                gaussian_kernel_matrix, sigmas=tf.constant(self.sigmas)
            )
            functools.update_wrapper(self.kernel_f, gaussian_kernel_matrix)

        self.mu_prior = tf.constant(mu_prior, dtype=tf.float32, shape=(size,))
        self.sigma_prior = tf.constant(sigma_prior, dtype=tf.float32, shape=(size,))

    def add_mmd_loss(self, z, z_prior):
        mmd = maximum_mean_discrepancy(z, z_prior, kernel=self.kernel_f)
        mmd = tf.maximum(1e-4, mmd) * self.mmd_coef
        self.add_loss(mmd)
        self.add_metric(mmd, "mean", "mmd_discrepancy")

    def add_kl_loss(self, z, mu, logsigma):
        kl = kl_divergence(z, mu, logsigma, self.mu_prior, self.sigma_prior)
        kl = tf.maximum(1e-4, kl) * self.kl_coef
        self.add_loss(kl)
        self.add_metric(kl, "mean", "kl_divergence")

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(logvar * 0.5) + mean

    def call(self, inputs):

        if self.use_mmd:
            z = self.z(inputs)
            z_prior = tf.random.normal(tf.shape(z))
            self.add_mmd_loss(z, z_prior)

        if self.use_kl:
            mean, logvar = tf.split(inputs, num_or_size_splits=2, axis=1)
            mu = self.mu_layer(mean)
            logsigma = self.sigma_layer(logvar)
            z = self.reparameterize(mu, logsigma)
            self.add_kl_loss(z, mu, logsigma)

        return z

    def get_config(self):
        base_config = super(VariationalLayer, self).get_config()
        config = {
            "use_kl": self.use_kl,
            "use_mmd": self.use_mmd,
        }

        return dict(list(base_config.items()) + list(config.items()))
