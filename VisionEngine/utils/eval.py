import tensorflow as tf
import math


def embed_images(x, model):
    outputs = [
        model.model.get_layer("variational_layer").output,
        model.model.get_layer("variational_layer_1").output,
        model.model.get_layer("variational_layer_2").output,
        model.model.get_layer("variational_layer_3").output,
    ]
    encoder = tf.keras.Model(model.model.inputs, outputs)
    return encoder.predict(x)


def reconstruct_images(x, model):
    return model.model.predict(x)


class LikeLihoodLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LikeLihoodLayer, self).__init__(**kwargs)
        self.model_input_shape = [256, 256, 3]

    def build(self, input_shape):
        super(LikeLihoodLayer, self).build(input_shape)

    def call(self, layer_inputs, **kwargs):
        inputs, outputs = layer_inputs
        mse = -tf.losses.mean_squared_error(inputs, outputs)
        out = 1.0 / (tf.sqrt(2.0 * math.pi)) * tf.exp(-0.5 * (mse) ** 2.0)
        return out

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = {}
        base_config = super(LikeLihoodLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def sample_likelihood(x, model):
    inputs = tf.keras.layers.Flatten()(model.model.input)
    outputs = tf.keras.layers.Flatten()(model.model.output)
    out = LikeLihoodLayer()([inputs, outputs])
    lh_model = tf.keras.Model(model.model.input, out)
    return lh_model.predict(x)
