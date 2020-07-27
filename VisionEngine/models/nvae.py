class conv_block(tf.keras.layers.Layer):
    def __init__(self, c, **kwargs):
        super(conv_block, self).__init__(**kwargs)
        self.c = c

    def build(self, input_shape):
        self.cb = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                self.c // 2,
                kernel_size=3,
                padding='same'
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(
                self.c,
                kernel_size=3,
                padding='same'
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('swish')
        ])

    def call(self, layer_inputs, **kwargs):
        return self.cb(layer_inputs)

    def comput_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'c': self.c,
        }
        base_config = \
            super(conv_block, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class encoder_cell(tf.keras.layers.Layer):
    def __init__(self, cs):
        super(encoder_cell, self).__init__(**kwargs)
        self.cs = cs

    def build(self, input_shape):
        self.ec = tf.keras.Sequential()
        for cs_ in self.cs:
            self.ec.add(self.conv_block(cs_))

    def call(self, layer_inputs, **kwargs):
        return self.ec(layer_inputs)

    def comput_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'cs': self.cs,
        }
        base_config = \
            super(encoder_cell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class upsample_block(tf.keras.layers.Layer):
    def __init__(self, c):
        super(upsample_block, self).__init__(**kwargs)
        self.c = c

    def build(self, input_shape):
        self.ub = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(
                self.c,
                kernel_size=3,
                stride=2
            ),
            tf.keras.layers.BatchNormalization()
        ])

    def call(self, layer_inputs, **kwargs):
        return self.ub(layer_inputs)

    def comput_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'c': self.c,
        }
        base_config = \
            super(upsample_block, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class decoder_cell(tf.keras.layers.Layer):
    def __init__(self, cs):
        super(decoder_cell, self).__init__(**kwargs)
        self.cs = cs

    def build(self, input_shape):
        self.dc = tf.keras.Sequential()
        for cs_ in self.cs:
            self.dc.add(self.upsample_block(cs_))

    def call(self, layer_inputs, **kwargs):
        return self.se(layer_inputs)

    def comput_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'cs': self.cs
        }
        base_config = \
            super(decoder_cell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class encoder_residual_cell(tf.keras.layers.Layer):
    def __init__(self, c):
        super(encoder_residual_cell, self).__init__(**kwargs)
        self.c = c

    def build(self, input_shape):
        self.erc = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(
                c,
                kernel_size=3,
                padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('swish'),
            tf.keras.layers.Conv2D(
                c,
                kernel_size=3,
                padding='same'),
            SqueezeExcite])

    def call(self, layer_inputs, **kwargs):
        return self.erc(layer_inputs)

    def comput_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'c': self.c
        }
        base_config = \
            super(encoder_residual_cell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class decoder_residual_cell(tf.keras.layers.Layer):
    def __init__(self, c, e):
        super(decoder_residual_cell, self).__init__(**kwargs)
        self.c = c
        self.e = e

    def build(self, input_shape):
        self.drc = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(
                c * e,
                kernel_size=1,
                padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('swish'),
            tf.keras.layers.DepthwiseConv2D(
                c * e,
                kernel_size=5,
                stride=1),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('swish'),
            tf.keras.layers.Conv2D(
                c,
                kernel_size=1,
                padding='same',
                use_bias=False,
                activation=None),
            tf.keras.layers.BatchNormalization(),
            SqueezeExcite(c)])

        def call(self, layer_inputs, **kwargs):
            return self.drc(layer_inputs)

        def comput_output_shape(self, input_shape):
            return input_shape

        def get_config(self):
            config = {
                'c': self.c,
                'e': self.e
            }
            base_config = \
                super(decoder_residual_cell, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))


class SqueezeExcite(tf.keras.layers.Layer):
    def __init__(self, c, r=16, **kwargs):
        super(SqueezeExcite, self).__init__(**kwargs)
        self.c = c
        self.r = r

    def build(self, input_shape):
        self.se = tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(self.c // self.r, bias=False),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(c, bias=False),
            tf.keras.layers.Activation('sigmoid')])

    def call(self, layer_inputs, **kwargs):
        return self.se(layer_inputs)

    def comput_output_shape(self, input_shape):
        return input_shape

    def et_config(self):
        config = {
            'c': self.c,
            'r': self.r
        }
        base_config = \
            super(SqueezeExcite, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class NouveauVAE(tf.keras.Model):

    def __init__(self, z_dim, input_dim):
        super(NouveauVAE, self).__init__()
        self.z_dim = z_dim
        self.input_dim = input_dim

        # encoder
        self.enc = tf.keras.Sequential()

        self.enc.add(tf.keras.layers.Input(shape=self.input_dim))

        encoder_stack = [
            encoder_cell([self.z_dim // 16, self.z_dim // 8]),
            encoder_cell([self.z_dim // 4, self.z_dim // 2]),
            encoder_cell([self.z_dim])
        ]
        encoder_res_stack = [
            encoder_residual_cell(self.z_dim // 8),
            encoder_residual_cell(self.z_dim // 2),
            encoder_residual_cell(self.z_dim)
        ]

        for e, r in zip(encoder_stack, encoder_res_stack):
            x = r()(e)
            self.enc.add(x)

        self.condition_x = tf.keras.layers.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Activation('swish'),
            tf.keras.layers.Conv2D(self.z_dim * 2)
        ])

        self.enc.add(self.condition_x)

        # decoder
        self.dec = tf.keras.Sequential()
        self.dec.add(tf.keras.layers.Input(shape=self.z_dim * 2))
        decoder_stack = [
            decoder_cell([z_dim // 2]),
            decoder_cell([z // 4, z // 8]),
            decoder_cell([z // 16, z // 32])
        ]
        decoder_res_stack = [
            decoder_residual_cell(z_dim // 2, e=1),
            decoder_residual_cell(z_dim // 8, e=2),
            decoder_residual_cell(z_dim // 32, e=4),
        ]

        for d, r in zip(decoder_stack, decoder_res_stack):
            x = r(d)
            self.dec.add(x)

        self.x_hat = tf.keras.layers.Conv2D(3, kernel_size=1)

        self.dec.add(self.x_hat)
    

    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.z_dim))
        return self.decode(eps, appy_sigmoid=True)

    def encode(self, x):
        mu, logvar = tf.split(self.encoder(x), num_or_size_of_splits=2, axis=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        eps = tf.random.normal(shape=mu.shape)
        return eps * tf.exp(logvar * .5) + mean
    
    def decode(self, z, apply_sigmoid=False):
        logits = self.decode(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits