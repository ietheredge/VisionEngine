import tensorflow as tf
import numpy as np
import numba


def make_perceptual_loss_model(input_shape, layers=[13]):
    loss_model = tf.keras.applications.VGG16(
        weights="imagenet", include_top=False, input_shape=input_shape
    )
    loss_model.trainable = False
    for layer in loss_model.layers:
        layer.trainable = False
    loss_layers = [loss_model.layers[i].output for i in layers]
    model = tf.keras.Model(loss_model.inputs, loss_layers)
    model.trainable = False
    return model


@numba.jit(nopython=True, parallel=True)
def calculate_perceptual_distances(X):
    norm_dists = np.zeros((len(X[0]), len(X)))
    for i in range(len(X)):
        for j in range(len(X[0])):
            norm_dists[j, i] = np.linalg.norm(X[i][j].flatten())
    return norm_dists
