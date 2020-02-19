# visualization utils
import tensorflow as tf
from tf.keras.layers import Flatten
import numpy as np
from PIL import Image
from itertools import product
import math
import os


model_folder = '../checkpoints/vlae_mmd_all'
image_output_folder = '../reports/figures/images'
plot_output_folder = '../reports/figures/panels'
n_latents = 4 
latent_size = 10


def make_rand_samples(model, n_samples=9, num_steps=300, mu=0, sigma=1):
    output_folder = os.path.join(image_output_folder, 'explore_latents/random_normal/frames')
    
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


def make_traversal_from_zeros(model, n_samples=1, num_steps=11):
    output_folder = os.path.join(image_output_folder, 'explore_latents/traversal')

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


def make_traversal_from_sample(model, z, n_samples=1, num_steps=11, sample_id=0):
    output_folder = os.path.join(image_output_folder, 'explore_latents/traversal')

    multipliers = np.linspace(-2,2,num=num_steps)
    encoded_sample = [z_i[sample_id] for z_i in z]

    for z_i in range(4):
        image_container = Image.new('RGB', (256*num_steps,256*latent_size))
        for z_i_j in range(latent_size):
            for s in range(num_steps):
                sample = [np.array([encoded_sample[0].numpy()]),
                      np.array([encoded_sample[1].numpy()]),
                      np.array([encoded_sample[2].numpy()]),
                      np.array([encoded_sample[3].numpy()])]
                
                sample[z_i][0][z_i_j] = multipliers[s]
                generated = model.get_layer('decoder').predict(sample, batch_size=1)
                generated = generated.reshape((256, 256,3))
                img = 255 * np.array(generated)
                img = img.astype(np.uint8)
                image_container.paste(Image.fromarray(img.astype('uint8')), (s*256, z_i_j*256))
        image_container.save(os.path.join(output_folder,'{}sample{}.jpg'.format(sample_id, z_i)))


def main():
    #TODO: Move these functions to separate dataprocessing util
    def custom_loss(x, xhat):
        return  .5 * tf.losses.mean_squared_error(tf.keras.layers.Flatten()(x), tf.keras.layers.Flatten()(xhat)) * np.prod([256,256,3])

    def sample_likelihood(x, x_hat):
        mse = - tf.losses.mean_squared_error(Flatten()(x), Flatten()(x_hat))
        return 1./(tf.sqrt(2.*math.pi))*tf.exp(-.5*(mse)**2.)

    def embed_images(x):
        x = vae.get_layer('encoder').predict(x, batch_size=10)
        return [vae.get_layer('z_1_latent')(x[0]), 
                vae.get_layer('z_2_latent')(x[1]),
                vae.get_layer('z_3_latent')(x[2]),
                vae.get_layer('z_4_latent')(x[3])]

    def reconstruct_images(z):
        return vae.get_layer('decoder').predict(z)

    vae = tf.keras.models.load_model(model_folder, custom_objects={'loss': custom_loss}, compile=False)
    vae.compile()




    with tf.device("GPU:0"):
        z = embed_images(real_images)
        x_hat = reconstruct_images(z)
        likelihood = sample_likelihood(real_images, x_hat)

    make_rand_samples(vae)
    make_traversal_from_zeros(vae)
    make_traversal_from_sample(vae, z, sample_id=230)  # sample ID 230 presented in manuscript
