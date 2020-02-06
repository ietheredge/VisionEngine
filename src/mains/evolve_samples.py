import numpy as np
import tensorflow as tf
import tqdm
import warnings
import os
import numba
from itertools import product


population_size = 1000
generations = 1000
n_latents = 4
latent_size = 10


def generate_fitness_surface(overwrite=False):
    warnings.warn("We are evaluating 100000 points in our high dimension space. This may take a while, even with a powerful GPU...")

    def custom_loss(x,xhat):
        return  .5 * tf.losses.mean_squared_error(Flatten()(x), Flatten()(xhat)) * np.prod(images[0].shape)
    model_folder = '../models/vlae_mmd_all'
    model = tf.keras.models.load_model(model_folder, custom_objects={'loss': custom_loss}, compile=False)
    model.compile()

    # evenly sample the latent space
    X = np.random.uniform(low=-4,high=4,size=(4, 100000, 10))
    with tf.device("GPU:0"):
        orange_min = tf.constant([0.9, 0.55, 0.])
        orange_max = tf.constant([1., 0.75, 0.1])
        black_min = tf.constant([0., 0., 0.])
        black_max = tf.constant([[0.2, 0.2, 0.2]])
        weights = [1., 1.]

        fitness = []
        for i in tqdm.tqdm(range(100000)):
            x_hat = model.get_layer('decoder').predict([np.expand_dims(X[0,i],axis=0),
                                                        np.expand_dims(X[1,i],axis=0),
                                                        np.expand_dims(X[2,i],axis=0),
                                                        np.expand_dims(X[3,i],axis=0)])
            orange_vals = tf.math.logical_and(
                    tf.math.greater(x_hat,orange_min),
                    tf.math.less(x_hat, orange_max))
            percent_orange = tf.math.divide(
                tf.reduce_sum(tf.cast(tf.reduce_all(
                        orange_vals,axis=(3)
                    ),dtype=tf.float32
                ),axis=(1,2)),
                np.product([256,256]))
            black_vals = tf.math.logical_and(
                    tf.math.greater(x_hat,black_min),
                    tf.math.less(x_hat, black_max))
            percent_black = tf.math.divide(
                tf.reduce_sum(tf.cast(tf.reduce_all(
                        black_vals,axis=(3)
                    ),dtype=tf.float32
                ),axis=(1,2)),
                np.product([256,256]))
            # fitness is just a simple weighted sum here
            fit = tf.math.reduce_sum([percent_orange*weights[0],percent_black*weights[1]],axis=0)
            fitness.append(fit)
        return X, np.array(fitness)


@numba.jit(nopython=True, parallel=True)
def fitness(parents, attribute_table, fitness_table):
    '''
    We define a simple fitness metric where the percent orange
    and percent black contribute to higher fitness.
    We use a lookup table of a predefined fitness landscape (for speed).
    '''
    fitness = []
    for i in range(len(parents)):
        fitness.append(np.argmin(np.sum(np.abs(attribute_table - parents[i]),axis=1)))

    return fitness
        

def selection(parents, fitness, persistence=0.5, temperature=0.2):
    '''
    perform the selection step on the next generation of parents
    '''
    p_dist = np.array(fitness/np.sum(fitness)).flatten()
    indexes = np.arange(len(parents))
    survivors = parents[np.random.choice(indexes, int(population_size*persistence), p=p_dist)]
    survivors = np.concatenate([survivors, parents[np.random.choice(indexes, int(population_size*temperature))]])
    return survivors


def mutate(child, mutation_rate=1, temperature=3):
    '''
    add mutations to offspring
    '''
    # add N random mutations to child with a given temperature
    # destabilizing 
    for _ in range(mutation_rate):
        z_i = np.random.choice(range(n_latents))
        z_i_j = np.random.choice(range(latent_size))
        child[z_i][0][z_i_j] = np.random.normal(loc=0, scale=temperature)
    # stabilizing 
    for _ in range(mutation_rate):
        z_i = np.random.choice(range(n_latents))
        z_i_j = np.random.choice(range(latent_size))
        child[z_i][0][z_i_j] = 0.

    return child


def crossing(parents):
    '''
    pass on alleles
    '''
    offspring = []
    for _ in range(int(population_size - len(parents))):

        # pick a couple of parents
        parent1 = parents[np.random.choice(np.arange(len(parents)))]
        parent2 = parents[np.random.choice(np.arange(len(parents)))]
        
        # randomly initialize child
        child = [
            np.random.multivariate_normal([0] * latent_size,np.diag([1] * latent_size), 1)
            ] * n_latents

        # randomly combine traits from each parent with equal probability
        locs = product(range(n_latents),range(latent_size))
        for z_i, z_i_j in locs:
            child[z_i][0][z_i_j] = np.random.choice([parent1[z_i][z_i_j], parent2[z_i][z_i_j]])
        
        child = mutate(child)
        offspring.append(np.array(child).reshape(4,10))

    return np.array(offspring)


def main():
    # start with an initial population
    parent_record = []
    parents = [
            np.random.multivariate_normal([0] * latent_size,np.diag([1] * latent_size), population_size)
            ] * n_latents
    
    
    # load our fitness surface
    attribute_table = X
    fitness_table = fit
    
    # reshape arrays
    parents = np.transpose(np.array(parents), (1,0,2))
    attribute_table = np.transpose(attribute_table, (1,0,2))
    # start the evolutionary process
    for _ in tqdm.tqdm(range(generations)):
        parent_fitness = fitness(parents.reshape(parents.shape[0],np.prod(parents.shape[1:])),
                                 attribute_table.reshape(attribute_table.shape[0],np.prod(attribute_table.shape[1:])),
                                 fitness_table)
        survivors = selection(parents, parent_fitness)
        offspring = crossing(survivors)  # also includes mutation
        parents = np.concatenate([survivors,offspring])
        parent_record.append(parents)