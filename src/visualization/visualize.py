# visualization utils
from PIL import Image
from itertools import product
import numpy as np
import os


class LatentExploration():
    def __init__(self, model, n_latents=4, latent_size=10, output_folder=None):
        self.model = model
        self.n_latents = n_latents
        self.latent_size = latent_size
        if output_folder is None:
            self.output_folder = '../reports/figures/images/explore_latents'
        else:
            self.output_folder = output_folder

    def make_rnorm_samples(self, n_samples=9, num_steps=300, mu=0, sigma=1, subfolder=None):
        if subfolder is None:
            output_folder = os.path.join(self.output_folder, 'random_normal/frames')
        else:
            output_folder = os.path.join(self.output_folder, subfolder)
        
        sample =  [
            np.random.multivariate_normal([mu] * self.latent_size,np.diag([sigma] * self.latent_size), n_samples)
            ] * self.n_latents
        for z in range(self.n_latents):
            for t in range(num_steps):
                sample[z] = np.random.multivariate_normal(
                    [mu] * self.latent_size, np.diag([sigma] * self.latent_size), n_samples)
                generated = self.model.get_layer('decoder').predict(sample, batch_size=10)
                generated = generated.reshape((n_samples, 256, 256,3))
                image_container = Image.new('RGB', (256*3,256*3))
                locs = list(product(range(int(np.sqrt(n_samples))),range(int(np.sqrt(n_samples)))))
                for i in range(n_samples):
                    img = generated[i]
                    j, k = locs[i]
                    img = 256 * np.array(img)
                    img = img.astype(np.uint8)
                    image_container.paste(Image.fromarray(img.astype('uint8')), (k*256, j*256))
                image_container.save(os.path.join(output_folder,'z{}_{:03d}.jpg'.format(z,t)))

    def make_traversal_samples(self, n_samples=1, num_steps=10, subfolder=None):
        if subfolder is None:
            output_folder = os.path.join(self.output_folder, 'traversal')
        else:
            output_folder = os.path.join(self.output_folder, subfolder)

        n_samples=1
        num_steps=11
        multipliers = np.linspace(-2,2,num=num_steps)
        # sample = [np.random.multivariate_normal([0] * latent_size, np.diag([0] * latent_size), 1)] * 4
        sample = [np.array([[0] * latent_size])] * 4
        sample_orig = sample.copy()
        for z_i in range(4):
            image_container = Image.new('RGB', (256*num_steps,256*latent_size))
            for z_i_j in range(latent_size):
                for s in range(num_steps):
                    sample = [np.array([[0] * latent_size]),
                            np.array([[0] * latent_size]),
                            np.array([[0] * latent_size]),
                            np.array([[0] * latent_size])]
                    
                    sample[z_i][0][z_i_j] = multipliers[s]
                    generated = vlae.get_layer('decoder').predict(sample, batch_size=1)
                    generated = generated.reshape((256, 256,3))
                    img = 255 * np.array(generated)
                    img = img.astype(np.uint8)
                    image_container.paste(Image.fromarray(img.astype('uint8')), (s*256, z_i_j*256))
            image_container.save(os.path.join(output_folder,'z{}.jpg'.format(z_i)))
