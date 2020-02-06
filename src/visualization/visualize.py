# visualization utils
import tensorflow as tf
import numpy as np
from PIL import Image
from itertools import product
import os

model_folder = '../models/vlae_mmd_all'
output_folder = '../reports/figures/images/explore_latents'
n_latents = 4 
latent_size = 10


def make_rnorm_frames(model, n_samples=9, num_steps=300, mu=0, sigma=1, subfolder=None):
    if subfolder is None:
        output_folder = os.path.join(output_folder, 'random_normal/frames')
    else:
        output_folder = os.path.join(output_folder, subfolder)
    
    sample =  [
        np.random.multivariate_normal([mu] * latent_size,np.diag([sigma] * latent_size), n_samples)
        ] * n_latents

    for z in range(n_latents):
        for t in range(num_steps):
            sample[z] = np.random.multivariate_normal(
                [mu] * latent_size, np.diag([sigma] * latent_size), n_samples)
            generated = model.get_layer('decoder').predict(sample, batch_size=10)
            generated = generated.reshape((n_samples, 256, 256,3))
            image_container = Image.new('RGB', (256*3,256*3))
            locs = list(product(range(int(np.sqrt(n_samples))),range(int(np.sqrt(n_samples)))))
            for i in range(n_samples):
                img = generated[i]
                j, k = locs[i]
                img = 255 * np.array(img)
                img = img.astype(np.uint8)
                image_container.paste(Image.fromarray(img.astype('uint8')), (k*256, j*256))
            image_container.save(os.path.join(output_folder,'z{}_{:03d}.jpg'.format(z,t)))


def make_traversal_samples(model, n_samples=1, num_steps=11, subfolder=None):
    if subfolder is None:
        output_folder = os.path.join(output_folder, 'traversal')
    else:
        output_folder = os.path.join(output_folder, subfolder)

    multipliers = np.linspace(-2,2,num=num_steps)

    for z_i in range(4):
        image_container = Image.new('RGB', (256*num_steps,256*latent_size))
        for z_i_j in range(latent_size):
            for s in range(num_steps):
                sample = [np.array([[0] * latent_size]),
                        np.array([[0] * latent_size]),
                        np.array([[0] * latent_size]),
                        np.array([[0] * latent_size])]
                
                sample[z_i][0][z_i_j] = multipliers[s]
                generated = model.get_layer('decoder').predict(sample, batch_size=1)
                generated = generated.reshape((256, 256,3))
                img = 255 * np.array(generated)
                img = img.astype(np.uint8)
                image_container.paste(Image.fromarray(img.astype('uint8')), (s*256, z_i_j*256))
        image_container.save(os.path.join(output_folder,'z{}.jpg'.format(z_i)))


def main():
    def custom_loss(x,xhat):
        return  .5 * tf.losses.mean_squared_error(tf.keras.layers.Flatten()(x), tf.keras.layers.Flatten()(xhat)) * np.prod([256,256,3])
    vae = tf.keras.models.load_model(model_folder, custom_objects={'loss': custom_loss}, compile=False)
    vae.compile()

    make_rnorm_frames(vae)
    make_traversal_samples(vae)
