import tensorflow as tf
import tensorflow_probability as tfp


class NormalVariational(tf.keras.layers.Layer):
    
    def __init__(self, size, mu_prior=0., sigma_prior=1., add_gmm=False, n_mixtures=None, add_kl=False, coef_kl=1.0, add_mmd=True, lambda_mmd=1.0, kernel_f=None, name=None):
        super().__init__(name=name)
        self.mu_layer = tf.keras.layers.Dense(size)
        self.sigma_layer = tf.keras.layers.Dense(size)
        self.mu_prior = tf.constant(mu_prior, dtype=tf.float32, shape=(size,))
        self.sigma_prior = tf.constant(sigma_prior, dtype=tf.float32, shape=(size,))

        self.add_kl = add_kl
        self.coef_kl = tf.Variable(coef_kl, trainable=False, name='coef_kl')

        self.add_mmd = add_mmd
        self.lambda_mmd = lambda_mmd
        if kernel_f is None:
            self.kernel_f = self._rbf
        else:
            self.kernel_f = kernel_f

        self.add_gmm = add_gmm
        self.n_mixtures = n_mixtures
        if self.n_mixtures:
            self.mu_layers = [tf.keras.layers.Dense(size) for i in range(n_mixture)]
            self.sigma_layers = [tf.keras.layers.Dense(size) for i in range(n_mixtures)]
            self.mu_priors = [tf.constant(mu_prior, dtype=tf.float32, shape=(size,)) for i in range(n_mixture)]
            self.sigma_priors = [tf.constant(sigma_prior, dtype=tf.float32, shape=(size,)) for i in range(n_mixture)]
            
    def _rbf(self, x, y):
        x_size = tf.shape(x)[0]
        y_size = tf.shape(y)[0]
        dim = tf.shape(x)[1]
        tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
        tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
        return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))

    def add_kl_divergence(self, q_mu, q_sigma, p_mu, p_sigma):
        r = q_mu - p_mu
        kl = tf.reduce_mean(self.coef_kl * tf.reduce_sum(tf.math.log(p_sigma) - tf.math.log(q_sigma) - .5 * (1. - (q_sigma**2 + r**2) / p_sigma**2), axis=1))
        self.add_loss(kl)
        self.add_metric(kl, 'mean', 'kl_divergence')

    def add_mm_discrepancy(self, z, z_prior):
        k_prior = self.kernel_f(z_prior, z_prior)
        k_post = self.kernel_f(z, z)
        k_prior_post = self.kernel_f(z_prior, z)
        mmd = tf.reduce_mean(k_prior) + tf.reduce_mean(k_post) - 2 * tf.reduce_mean(k_prior_post)
        mmd = tf.multiply(self.lambda_mmd,  mmd, name='mmd')
        self.add_loss(mmd)
        self.add_metric(mmd, 'mean', 'mmd_discrepancy')

    def add_gmm_log_prob(self, z)
        gmm_loss = - tf.reduce_mean(self.mixture_model.log_prob(z))
        self.add_loss(gmm_loss)
        self.add_metric(gmm_loss, 'mean', 'gmm_log_prob')

    def call(self, inputs):
        if self.add_mmd:
            mu = self.mu_layer(inputs)
            log_sigma =  self.sigma_layer(inputs)
            sigma_square = tf.exp(log_sigma)
            z = mu + sigma_square * tf.random.normal(tf.shape(sigma_square))
            z_prior = tfp.distributions.MultivariateNormalDiag(self.mu_prior, self.sigma_prior).sample(tf.shape(z)[0])
            self.add_mm_discrepancy(z, z_prior)

        if self.add_kl:
            mu = self.mu_layer(inputs)
            log_sigma =  self.sigma_layer(inputs)
            sigma_square = tf.exp(log_sigma)
            self.add_kl_divergence(mu, sigma_square, self.mu_prior, self.sigma_prior)
        
        if self.add_gmm:
            probs = tf.Variable(shape=[n_mixtures], dtype=tf.float32, name='mixture_probs')
            probs = tf.nn.softmax(probs)
            scale_fn = lambda name: tf.nn.softplus(tf.Variable(np.ones([size], dtype='float32'), name=name))
            self.mixture_model = tf.distributions.MixtureSameFamily(
                mixture_distribution=tf.distributions.Categorical(probs=probs),
                components_distribution=tf.distributions.MultivariateNormalDiag(
                        loc=[tf.Variabel(shape=[size], name='loc{}'.format(i)) for i in range(n_mixtures)]
                        scale_identity_multiplier=[scale_fn('scale{}'.format(i)) for i in range(n_mixtures)]
                    )
            )
            mus = [mu_layer(inputs) for mu_layer in self.mu_layers]
            log_sigmas = [sigma_layer(inputs) for sigma_layer in self.sigma_layers]
            sigma_squares = tf.exp(log_sigma)
            z = mus + sigma_squares * tf.random.normal(tf.shape(sigma_square))
            self.add_gmm_log_prob(z)

        return z

    def get_config(self):
        base_config = super(NormalVariational, self).get_config()
        config = {
            'add_kl': self.add_kl,
            'add_mmd': self.add_mmd,
            'lambda_mmd': self.lambda_mmd,
            'use_gmm': self.use_gmm,
            'n_mixtures': self.n_mixtures,
        }
        return dict(list(base_config.items()) + list(config.items()))

