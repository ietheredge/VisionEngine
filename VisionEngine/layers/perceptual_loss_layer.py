"""
Copyright (c) 2020 R. Ian Etheredge All rights reserved.

This work is licensed under the terms of the MIT license.
For a copy, see <https://opensource.org/licenses/MIT>.
"""
import tensorflow as tf


class PerceptualLossLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        perceptual_loss_model,
        pereceptual_loss_layers,
        perceptual_loss_layer_weights,
        model_input_shape,
        name,
        **kwargs
    ):
        super(PerceptualLossLayer, self).__init__(**kwargs)
        self.loss_model_type = perceptual_loss_model
        self.layers = pereceptual_loss_layers
        self.loss_layer_weights = perceptual_loss_layer_weights
        self.model_input_shape = [256, 256, 3]
        self.n_layers = len(pereceptual_loss_layers)

    def build(self, input_shape):
        if self.loss_model_type == "vgg":
            self.loss_model_ = tf.keras.applications.VGG16(
                weights="imagenet",
                include_top=False,
                input_shape=self.model_input_shape,
            )
            self.loss_model_.trainable = False

            for layer in self.loss_model_.layers:
                layer.trainable = False

            self.loss_layers = [
                self.loss_model_.get_layer(name).output for name in self.layers
            ]

            self.loss_model = tf.keras.Model(
                [self.loss_model_.input],
                self.loss_layers,
            )
            self.loss_model.trainable = False

        else:
            raise NotImplementedError
        super(PerceptualLossLayer, self).build(input_shape)

    def call(self, layer_inputs, **kwargs):
        y_true = layer_inputs[0]
        y_pred = layer_inputs[1]

        y_true = tf.keras.applications.vgg19.preprocess_input(y_true * 255.0)
        y_pred = tf.keras.applications.vgg19.preprocess_input(y_pred * 255.0)

        sample = self.loss_model(y_true)
        reconstruction = self.loss_model(y_pred)

        self.sample_ = [self.gram_matrix(sample_output) for sample_output in sample]
        self.reconstruction_ = [
            self.gram_matrix(reconstruction_output)
            for reconstruction_output in reconstruction
        ]

        perceptual_loss = tf.add_n(
            [
                tf.reduce_mean((self.reconstruction_[name] - self.sample_[name]) ** 2)
                for name, _ in enumerate(self.reconstruction_)
            ]
        )
        perceptual_loss *= self.loss_layer_weights / self.n_layers

        self.add_loss(perceptual_loss)
        self.add_metric(perceptual_loss, "mean", "perceptual_loss")

        return [layer_inputs[0], layer_inputs[1]]

    def compute_output_shape(self, input_shape):
        return input_shape

    @staticmethod
    def gram_matrix(input_tensor):
        result = tf.linalg.einsum("bijc,bijd->bcd", input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        return result / (num_locations)

    def get_config(self):
        config = {
            "perceptual_loss_model": self.loss_model_type,
            "pereceptual_loss_layers": self.layers,
            "perceptual_loss_layer_weights": self.loss_layer_weights,
            "model_input_shape": self.model_input_shape,
        }
        base_config = super(PerceptualLossLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
