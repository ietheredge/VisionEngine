import tensorflow as tf
import numpy as np


class SIRENFirstLayerInitializer(tf.keras.initializers.Initializer):
    def __init__(self, scale=1.0, seed=None):
        super().__init__()
        self.scale = scale
        self.seed = seed

    def __call__(self, shape, dtype=tf.float32):
        fan_in = shape[0]
        fan_out = shape[1]
        limit = self.scale / max(1.0, float(fan_in))
        return tf.random.uniform(shape, -1, 1, seed=self.seed)

    def get_config(self):
        base_config = super().get_config()
        config = {
            'scale': self.scale,
            'seed': self.seed
        }
        return dict(list(base_config.items()) + list(config.items()))


class SIRENInitializer(tf.keras.initializers.VarianceScaling):
    def __init__(self, w0: float = 1.0, c: float = 6.0, seed: int = None):
        # Uniform variance scaler multiplies by 3.0 for limits, so scale down here to compensate
        self.w0 = w0
        self.c = c
        scale = c / (3.0 * w0 * w0)
        super(SIRENInitializer, self).__init__(scale=scale, mode='fan_in', distribution='uniform', seed=seed)

    def get_config(self):
        base_config = super().get_config()
        config = {
            'w0': self.w0,
            'c': self.c
        }
        return dict(list(base_config.items()) + list(config.items()))


class Sine(tf.keras.layers.Layer):
    def __init__(self, w0: float = 1.0, **kwargs):
        """
        Sine activation function with w0 scaling support.
        Args:
            w0: w0 in the activation step `act(x; w0) = sin(w0 * x)`
        """
        super(Sine, self).__init__(**kwargs)
        self.w0 = w0

    def call(self, inputs, **kwargs):
        return tf.sin(self.w0 * inputs)
    
    def get_config(self):
        base_config = super().get_config()
        config = {
            'w0': self.w0,
        }
        return dict(list(base_config.items()) + list(config.items()))


class DenseSiren(tf.keras.layers.Dense):
    """
    Dense Siren Layer
    """

    def __init__(self, units,
                 w0: float = 1.0,
                 c: float = 6.0,
                 activation='sine',
                 use_bias=True,
                 kernel_initializer='siren_uniform',
                 bias_initializer='siren_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        if activation == 'sine':
            activation = Sine(w0=w0)

        if kernel_initializer == 'siren_uniform':
            kernel_initializer = SIRENInitializer(w0=w0, c=c)

        if bias_initializer == 'siren_uniform':
            bias_initializer = SIRENInitializer(w0=w0, c=c)

        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.supports_masking = True

        super().__init__(
            units=units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        
        def get_config(self):
            base_config = super(DenseSiren, self).get_config()
            config = {
                'w0': self.w0,
                'c': self.c
            }
            return dict(list(base_config.items()) + list(config.items()))


class DenseTiedSiren(tf.keras.layers.Dense):
    """
    Tied Dense Siren Layer
    """

    def __init__(self, units,
                 w0: float = 1.0,
                 c: float = 6.0,
                 activation='sine',
                 use_bias=True,
                 kernel_initializer='siren_uniform',
                 bias_initializer='siren_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 tied_to=None,
                 **kwargs):
        self.tied_to = tied_to
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        if activation == 'sine':
            activation = Sine(w0=w0)

        if kernel_initializer == 'siren_uniform':
            kernel_initializer = SIRENInitializer(w0=w0, c=c)

        if bias_initializer == 'siren_uniform':
            bias_initializer = SIRENInitializer(w0=w0, c=c)

        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=2)
        self.supports_masking = True

        super().__init__(
            units=units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
                
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        if self.tied_to is not None:
            self.kernel = tf.transpose(self.tied_to.kernel)
            self._non_trainable_weights.append(self.kernel)
        else:
            self.kernel = self.add_weight(shape=(input_dim, self.units),
                                          initializer=self.kernel_initializer,
                                          name='kernel',
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def call(self, inputs):
        output = tf.keras.backend.dot(inputs, self.kernel)
        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_config(self):
        base_config = super(DenseTiedSiren, self).get_config()
        config = {
            'w0': self.w0,
            'c': self.c
        }
        return dict(list(base_config.items()) + list(config.items()))


class DenseTied(tf.keras.layers.Dense):
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 tied_to=None,
                 **kwargs):
        self.tied_to = tied_to
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
            
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=2)
        self.supports_masking = True

        super().__init__(
            units=units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
                
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        if self.tied_to is not None:
            self.kernel = tf.keras.backend.transpose(self.tied_to.kernel)
            self._non_trainable_weights.append(self.kernel)
        else:
            self.kernel = self.add_weight(shape=(input_dim, self.units),
                                          initializer=self.kernel_initializer,
                                          name='kernel',
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def call(self, inputs):
        output = tf.keras.backend.dot(inputs, self.kernel)
        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output


class WeightsOrthogonalityConstraint(tf.keras.constraints.Constraint):
    def __init__(self, encoding_dim, weightage: float = 1.0, axis = 0):
        self.encoding_dim = encoding_dim
        self.weightage = weightage
        self.axis = axis

    def weights_orthogonality(self, w):
        if(self.axis==1):
            w = tf.keras.backend.transpose(w)
        if(self.encoding_dim > 1):
            m = tf.keras.backend.dot(tf.keras.backend.transpose(w), w) - tf.eye(self.encoding_dim)
            return self.weightage * tf.sqrt(tf.keras.backend.sum(tf.square(m)))
        else:
            m = tf.keras.backend.sum(w ** 2) - 1.
            return m

    def __call__(self, w):
        return self.weights_orthogonality(w)
    

class UncorrelatedFeaturesConstraint(tf.keras.constraints.Constraint):
    def __init__(self, encoding_dim, weightage: float = 1.0):
        self.encoding_dim = encoding_dim
        self.weightage = weightage

    def get_covariance(self, x):
        x_centered_list = []

        for i in range(self.encoding_dim):
            x_centered_list.append(x[:, i] - tf.keras.backend.mean(x[:, i]))

        x_centered = tf.stack(x_centered_list)
        covariance = tf.keras.backend.dot(x_centered,tf.keras.backend.transpose(x_centered)) / \
            tf.cast(x_centered.get_shape()[0], tf.float32)

        return covariance

    # Constraint penalty
    def uncorrelated_feature(self, x):
        if(self.encoding_dim <= 1):
            return 0.0
        else:
            output = tf.keras.backend.sum(tf.square(
                self.covariance - tf.math.multiply(self.covariance, tf.eye(self.encoding_dim))))
            return output

    def __call__(self, x):
        self.covariance = self.get_covariance(x)
        return self.weightage * self.uncorrelated_feature(x)

tf.keras.utils.get_custom_objects().update({
    'sine': Sine,
    'siren_uniform': SIRENInitializer,
    'siren_first_uniform': SIRENFirstLayerInitializer,
    'uncorrelated_features': UncorrelatedFeaturesConstraint,
    'weights_orthogonality': WeightsOrthogonalityConstraint
})