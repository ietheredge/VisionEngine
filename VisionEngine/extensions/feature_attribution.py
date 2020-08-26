import tensorflow as tf


def interpolate_latentvar(Z, H, z_i, alphas, zdim=10):
    mods = []
    for h in range(len(Z)):
        
        if h == H:
            z = Z[h]
            mod = tf.concat([
                tf.repeat(
                    [z], 10, axis=0)[:,:z_i],
                alphas[:, tf.newaxis],
                tf.repeat(
                    [z], 10, axis=0)[:, z_i:-1]], 1)
            mods.append(mod)
        else:
            z = Z[h]
            mod = tf.repeat([z], 10, axis=0)
            mods.append(mod)

    return mods

def compute_gradients(latent_vars, baseline, model):
    with tf.GradientTape() as tape:
        tape.watch(latent_vars)
        logits = model.decoder([latent_vars[0], latent_vars[1], latent_vars[2], latent_vars[3]])
        # logits = tf.nn.sigmoid(logits) # alt tf.nn.sigmoid, tf.nn.tanh
        mse = (logits - baseline) ** 2
        loss = tf.reduce_mean(mse)
        # images = tf.keras.layers.subtract([logits, baseline]) # alt tf.nn.sigmoid, tf.nn.tanh
    return tape.gradient(loss, logits)


def integral_approximation(gradients):
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    integrated_gradients = tf.math.reduce_sum(grads, axis=0)
    return integrated_gradients

def integrated_gradients(encoding, baseline, model, H=0, z_i=0, m_steps=1500, batch_size=10, ls_start=0, ls_stop=1):

    # Generate traversal steps
    traversal_steps = tf.linspace(start=ls_start, stop=ls_stop, num=m_steps)
    

    # Accumulate gradients across batches
    integrated_gradients = 0.0

    # Batch traversals
    ds = tf.data.Dataset.from_tensor_slices(traversal_steps).batch(batch_size)

    for batch in ds:
        batch_interpolated_inputs = interpolate_latentvar(Z=encoding, H=H, z_i=z_i, alphas=batch)

        batch_gradients = compute_gradients(batch_interpolated_inputs, baseline, model)

        integrated_gradients += integral_approximation(gradients=batch_gradients)
    
    return tf.math.abs(ls_stop-ls_start) * integrated_gradients

