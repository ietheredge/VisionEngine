import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from itertools import product
import math
import os
import tqdm
import tensorflow as tf
from pathlib import Path

plt.rcParams["pdf.use14corefonts"] = True


def imscatter(x, y, image, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    try:
        image = plt.imread(image)
    except TypeError:
        # Likely already an array...
        pass
    x, y = np.atleast_1d(x, y)
    artists = []
    for i, (x0, y0) in enumerate(zip(x, y)):
        im = OffsetImage(image[i], zoom=zoom, rasterized=True)
        ab = AnnotationBbox(im, (x0, y0), xycoords="data", frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    ax.grid(False)
    return artists


def plot_im(img, config):
    if config.model.last_activation == "tanh":
        img * 0.5 + 0.5
        return img
    else:
        return img


def make_rand_samples(
    model,
    image_output_folder,
    n_latents=4,
    latent_size=10,
    n_samples=9,
    num_steps=300,
    mu=0,
    sigma=1,
):
    output_folder = os.path.join(
        image_output_folder, "explore_latents/random_normal/frames"
    )

    sample = [
        np.random.multivariate_normal(
            [mu] * latent_size, np.diag([sigma] * latent_size), n_samples
        )
    ] * n_latents

    for z in range(n_latents):
        for t in range(num_steps):
            sample[z] = np.random.multivariate_normal(
                [mu] * latent_size, np.diag([sigma] * latent_size), n_samples
            )
            generated = model.get_layer("decoder").predict(sample, batch_size=10)
            generated = generated.reshape((n_samples, 256, 256, 3))
            image_container = Image.new("RGB", (256 * 3, 256 * 3))
            locs = list(
                product(range(int(np.sqrt(n_samples))), range(int(np.sqrt(n_samples))))
            )
            for i in range(n_samples):
                img = generated[i]
                j, k = locs[i]
                img = 255 * np.array(img)
                img = img.astype(np.uint8)
                image_container.paste(
                    Image.fromarray(img.astype("uint8")), (k * 256, j * 256)
                )
            image_container.save(
                os.path.join(output_folder, "z{}_{:03d}.jpg".format(z, t))
            )


def make_traversal_from_zeros(
    model, image_output_folder, latent_size=10, n_samples=1, num_steps=11
):
    output_folder = os.path.join(image_output_folder, "explore_latents/traversal")

    multipliers = np.linspace(-2, 2, num=num_steps)

    for z_i in range(4):
        image_container = Image.new("RGB", (256 * num_steps, 256 * latent_size))
        for z_i_j in range(latent_size):
            for s in range(num_steps):
                sample = [
                    np.array([[0] * latent_size]),
                    np.array([[0] * latent_size]),
                    np.array([[0] * latent_size]),
                    np.array([[0] * latent_size]),
                ]

                sample[z_i][0][z_i_j] = multipliers[s]
                generated = model.get_layer("decoder").predict(sample, batch_size=1)
                generated = generated.reshape((256, 256, 3))
                img = 255 * np.array(generated)
                img = img.astype(np.uint8)
                image_container.paste(
                    Image.fromarray(img.astype("uint8")), (s * 256, z_i_j * 256)
                )
        image_container.save(os.path.join(output_folder, "z{}.jpg".format(z_i)))


def make_traversal_from_sample(
    model,
    z,
    image_output_folder,
    latent_size=10,
    n_samples=1,
    num_steps=11,
    sample_id=0,
):
    output_folder = os.path.join(image_output_folder, "explore_latents/traversal")

    multipliers = np.linspace(-2, 2, num=num_steps)
    encoded_sample = [z_i[sample_id] for z_i in z]

    for z_i in range(4):
        image_container = Image.new("RGB", (256 * num_steps, 256 * latent_size))
        for z_i_j in range(latent_size):
            for s in range(num_steps):
                sample = [
                    np.array([encoded_sample[0].numpy()]),
                    np.array([encoded_sample[1].numpy()]),
                    np.array([encoded_sample[2].numpy()]),
                    np.array([encoded_sample[3].numpy()]),
                ]

                sample[z_i][0][z_i_j] = multipliers[s]
                generated = model.get_layer("decoder").predict(sample, batch_size=1)
                generated = generated.reshape((256, 256, 3))
                img = 255 * np.array(generated)
                img = img.astype(np.uint8)
                image_container.paste(
                    Image.fromarray(img.astype("uint8")), (s * 256, z_i_j * 256)
                )
        image_container.save(
            os.path.join(output_folder, "{}sample{}.jpg".format(sample_id, z_i))
        )


def plot_img_attributions(
    image, recon_img, attribution_mask, H=0, z_i=0, cmap=None, overlay_alpha=0.4
):

    fig, axs = plt.subplots(nrows=1, ncols=3, squeeze=False, figsize=(12, 4))

    axs[0, 0].set_title("Baseline Output")
    axs[0, 0].imshow(image, rasterized=True)
    axs[0, 0].axis("off")

    axs[0, 1].set_title(f"Feature Attribution: {H}, {z_i}")
    am = axs[0, 1].imshow(attribution_mask, cmap=cmap, rasterized=True)
    axs[0, 1].imshow(image, alpha=overlay_alpha, rasterized=True)
    fig.colorbar(am, ax=axs[0, 1])
    axs[0, 1].axis("off")

    axs[0, 2].set_title(f"Interpolated Output: {H}, {z_i}")
    axs[0, 2].imshow(recon_img, rasterized=True)
    axs[0, 2].axis("off")

    plt.tight_layout()
    return fig


def visualize_generation(generation, parent_record, BATCH_SIZE=1):
    i = 5
    j = i
    image_container = Image.new("RGB", (256 * i, 256 * j))
    x_hat = []
    subsample = parent_record[generation][: i * j]

    ds = (
        tf.data.Dataset.from_tensor_slices(subsample)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    for batch in tqdm.tqdm_notebook(ds, desc="generating"):
        batch = tf.reshape(batch, (4, BATCH_SIZE, 10))
        x_hat.extend(model.decoder([batch[0], batch[1], batch[2], batch[3]]))
    x_hat = tf.reshape(tf.Variable(x_hat), (i, j, 256, 256, 3))

    for (k, l) in tqdm.tqdm_notebook(product(range(i), repeat=2), desc="plotting"):
        sample = x_hat.numpy()[k, l, :]
        img = 255 * sample
        img = img.astype(np.uint8)
        image_container.paste(Image.fromarray(img.astype("uint8")), (k * 256, l * 256))
