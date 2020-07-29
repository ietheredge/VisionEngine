# embedd images
# z = q(x)

# perform traversal for each image
# baseline = {z | z_i = 0}
# for each latent variable (min, 0], [0, max)
min_zs = tf.math.minimum(Z)
max_zs = tf.math.maximum(Z)
min_alphas = tf.linspace(start=0., stop=min_zs, num=50)
max_alphas = tf.linspace(start=0., stop=max_zs, num=50)

def interpolate_images(baseline,
                       image,
                       alphas):
  alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
  baseline_x = tf.expand_dims(baseline, axis=0)
  input_x = tf.expand_dims(image, axis=0)
  delta = input_x - baseline_x
  images = baseline_x +  alphas_x * delta
  return images
  
def compute_gradients(images, target_class_idx):
  with tf.GradientTape() as tape:
    tape.watch(images)
    logits = model(images)
    probs = tf.nn.softmax(logits, axis=-1)[:, target_class_idx]
  return tape.gradient(probs, images)

def integral_approximation(gradients):
  # riemann_trapezoidal
  grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
  integrated_gradients = tf.math.reduce_mean(grads, axis=0)
  return integrated_gradients


@tf.function
def integrated_gradients(baseline,
                         image,
                         target_class_idx,
                         m_steps=300,
                         batch_size=32,
                         stp=1):
  # 1. Generate alphas
  alphas = tf.linspace(start=0.0, stop=stp, num=m_steps)

  # Accumulate gradients across batches
  integrated_gradients = 0.0

  # Batch alpha images
  ds = tf.data.Dataset.from_tensor_slices(alphas).batch(batch_size)

  for batch in ds:

    # 2. Generate interpolated images
    batch_interpolated_inputs = interpolate_images(baseline=baseline,
                                                   image=image,
                                                   alphas=batch)

    # 3. Compute gradients between model outputs and interpolated inputs
    batch_gradients = compute_gradients(images=batch_interpolated_inputs,
                                        target_class_idx=target_class_idx)

    # 4. Average integral approximation. Summing integrated gradients across batches.
    integrated_gradients += integral_approximation(gradients=batch_gradients)

  # 5. Scale integrated gradients with respect to input
  scaled_integrated_gradients = (image - baseline) * integrated_gradients
  return scaled_integrated_gradients