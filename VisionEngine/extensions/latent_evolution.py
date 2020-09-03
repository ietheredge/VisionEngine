import tensorflow as tf
import tqdm


def evolve_population(
    parents,
    model,
    parents_orig,
    temperature=0.2,
    persistence=0.5,
    mutation_per=0.1,
    N_GENERATIONS=500,
    POPULATION_SIZE=1000,
    BATCH_SIZE=100,
):

    parent_record = []

    parent_record.append(parents)  # record the starting generation

    orange_min = tf.constant([0.9, 0.55, 0.0])
    orange_min = tf.stack(
        [
            tf.fill((BATCH_SIZE, 256, 256), orange_min[0]),
            tf.fill((BATCH_SIZE, 256, 256), orange_min[1]),
            tf.fill((BATCH_SIZE, 256, 256), orange_min[2]),
        ],
        axis=-1,
    )
    orange_max = tf.constant([1.0, 0.75, 0.1])
    orange_max = tf.stack(
        [
            tf.fill((BATCH_SIZE, 256, 256), orange_max[0]),
            tf.fill((BATCH_SIZE, 256, 256), orange_max[1]),
            tf.fill((BATCH_SIZE, 256, 256), orange_max[2]),
        ],
        axis=-1,
    )
    black_min = tf.constant([0.0, 0.0, 0.0, 0.8])
    black_min = tf.stack(
        [
            tf.fill((BATCH_SIZE, 256, 256), black_min[0]),
            tf.fill((BATCH_SIZE, 256, 256), black_min[1]),
            tf.fill((BATCH_SIZE, 256, 256), black_min[2]),
        ],
        axis=-1,
    )
    black_max = tf.constant([0.2, 0.2, 0.2])
    black_max = tf.stack(
        [
            tf.fill((BATCH_SIZE, 256, 256), black_max[0]),
            tf.fill((BATCH_SIZE, 256, 256), black_max[1]),
            tf.fill((BATCH_SIZE, 256, 256), black_max[2]),
        ],
        axis=-1,
    )

    weights = [1.0, 1.0]

    for _ in tqdm.tqdm_notebook(range(N_GENERATIONS), desc="generation"):
        fitness = []
        ds = (
            tf.data.Dataset.from_tensor_slices(parents)
            .batch(BATCH_SIZE)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )
        for batch in tqdm.tqdm_notebook(ds, leave=False, desc="fitness loop"):
            batch = tf.reshape(batch, (4, BATCH_SIZE, 10))
            x_hat = model.decoder([batch[0], batch[1], batch[2], batch[3]])
            orange_vals = tf.math.logical_and(
                tf.math.greater(x_hat, orange_min), tf.math.less(x_hat, orange_max)
            )
            percent_orange = tf.math.divide(
                tf.reduce_sum(
                    tf.cast(tf.reduce_all(orange_vals, axis=(3)), dtype=tf.float32),
                    axis=(1, 2),
                ),
                256 * 256,
            )
            black_vals = tf.math.logical_and(
                tf.math.greater(x_hat, black_min), tf.math.less(x_hat, black_max)
            )
            percent_black = tf.math.divide(
                tf.reduce_sum(
                    tf.cast(tf.reduce_all(black_vals, axis=(3)), dtype=tf.float32),
                    axis=(1, 2),
                ),
                256 * 256,
            )
            # fitness is just a simple weighted sum here
            fit = tf.math.reduce_sum(
                [percent_orange * weights[0], percent_black * weights[1]], axis=0
            )

            rand_subsample = tf.random.uniform(
                [
                    BATCH_SIZE,
                ],
                minval=0,
                maxval=len(parents_orig),
                dtype=tf.dtypes.int32,
            )
            parents_orig_batch = tf.gather(parents_orig, rand_subsample)
            loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(x_hat, parents_orig_batch),
                axis=(1, 2),
            )
            fitness.extend(fit * tf.math.exp(-loss * 2))

        ####### survivors
        rank = tf.argsort(fitness, direction="DESCENDING")
        parents = tf.gather(parents, rank)

        indexes = tf.range(len(parents))

        #         indices_p = np.random.choice(indexes, int(POPULATION_SIZE*persistence), p=p_dist)
        #         indices_p = [[i.numpy()] * p_dist[i].numpy() for i in indexes]
        #         indices_p = [item for sublist in indices_p for item in sublist]
        #         indices_p = list(tf.random.shuffle(indices_p).numpy())[:int(POPULATION_SIZE*persistence)]
        survivors_p = tf.gather(parents, indexes[: int(POPULATION_SIZE * persistence)])

        #         indices_t = np.random.choice(indexes, int(POPULATION_SIZE*temperature))
        indices_t = list(tf.random.shuffle(indexes).numpy())[
            : int(POPULATION_SIZE * temperature)
        ]
        survivors_t = tf.gather(parents, indices_t)

        survivors = tf.concat([survivors_p, survivors_t], axis=0)

        ####### offspring
        # pick a couple of parents
        parent1 = [
            survivors[i]
            for i in tf.random.uniform(
                [
                    POPULATION_SIZE - len(survivors),
                ],
                minval=0,
                maxval=len(survivors),
                dtype=tf.dtypes.int32,
            )
        ]

        parent2 = [
            survivors[i]
            for i in tf.random.uniform(
                [
                    POPULATION_SIZE - len(survivors),
                ],
                minval=0,
                maxval=len(survivors),
                dtype=tf.dtypes.int32,
            )
        ]

        # randomly combine traits from each parent with equal probability
        mask = tf.dtypes.cast(
            tf.random.uniform(
                [POPULATION_SIZE - len(survivors), 4, 10],
                minval=0,
                maxval=2,
                dtype=tf.dtypes.int32,
            ),
            tf.bool,
        )

        children = tf.where(mask, parent1, parent2)

        values = tf.reduce_mean(
            tf.stack(
                [
                    parent1[int((POPULATION_SIZE - len(survivors)) * 0.8) :],
                    parent2[int((POPULATION_SIZE - len(survivors)) * 0.8) :],
                ]
            ),
            axis=0,
        )
        ix = list(tf.range(int((POPULATION_SIZE - len(survivors)) * 0.2)).numpy())
        indices = tf.stack(
            [
                tf.constant(
                    [
                        i,
                    ]
                )
                for i in ix
            ]
        )
        children = tf.tensor_scatter_nd_update(children, indices, values)

        #         children = tf.concat([children_c, children_x])
        #         children = tf.where(mask, parent1, parent2)

        # randomly mutate some loci
        ix = list(tf.random.shuffle([i for i in tf.range(len(children))]).numpy())[
            : int(len(children) * mutation_per)
        ]
        indices = tf.stack(
            [
                tf.constant(
                    [
                        [
                            i,
                            tf.random.uniform(
                                [
                                    1,
                                ],
                                minval=0,
                                maxval=5,
                                dtype=tf.dtypes.int32,
                            ).numpy()[0],
                            tf.random.uniform(
                                [
                                    1,
                                ],
                                minval=0,
                                maxval=11,
                                dtype=tf.dtypes.int32,
                            ).numpy()[0],
                        ]
                    ]
                )
                for i in ix
            ]
        )

        values = tf.stack(
            [
                tf.random.uniform(
                    [
                        1,
                    ],
                    minval=tf.math.reduce_mean(parents)
                    - (tf.math.reduce_std(parents) / 2),
                    maxval=tf.math.reduce_mean(parents)
                    + (tf.math.reduce_std(parents) / 2),
                    dtype=tf.dtypes.float32,
                )
                for i in ix
            ]
        )
        #         values = tf.stack([
        #             [tf.math.reduce_mean(parents)]
        #             for i in ix
        #         ])

        children = tf.tensor_scatter_nd_update(children, indices, values)

        parents = tf.concat([survivors, children], axis=0)

        parent_record.append(parents)

    return parent_record
