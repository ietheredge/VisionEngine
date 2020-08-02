from VisionEngine.layers.variational_layer import VariationalLayer
import functools
import tensorflow as tf

class MaximumMeanDiscrepancyTest(tf.test.TestCase):

    def test_mmd_is_zero_when_inputs_are_same(self):
        with self.test_session():
            x = tf.random.uniform((2, 3), seed=42)
            # TODO@ietheredge #5 partial does not pass self attribute need a different way to test
            kernel = functools.partial(
                VariationalLayer.gaussian_kernel_matrix,
                sigmas=tf.constant([1.]))
            functools.update_wrapper(kernel, VariationalLayer.gaussian_kernel_matrix)
        self.assertEquals(0, VariationalLayer.maximum_mean_discrepancy(x, x, kernel).eval()) 

if __name__ == '__main__':
    tf.test.main()