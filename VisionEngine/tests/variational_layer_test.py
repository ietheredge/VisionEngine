from VisionEngine.layers.variational_layer import gaussian_kernel_matrix
from VisionEngine.layers.variational_layer import maximum_mean_discrepancy

import functools

import tensorflow as tf

class MaximumMeanDiscrepancyTest(tf.test.TestCase):

    def test_mmd_is_zero_when_inputs_are_same(self):
        with self.test_session():
            x = tf.random.uniform((2, 3), seed=42)
            kernel = functools.partial(
                gaussian_kernel_matrix,
                sigmas=tf.constant([1.]))

            functools.update_wrapper(kernel, gaussian_kernel_matrix)

        self.assertEqual(0, maximum_mean_discrepancy(x, x, kernel).numpy()) 

if __name__ == '__main__':
    tf.test.main()