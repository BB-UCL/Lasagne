import numpy as np
import theano.tensor as T

from .. import init
from .. import nonlinearities
from ..utils import th_fx, fetch_pre_activation
from ..skfgn import get_fuse_bias
from . import helper

from .base import Layer


__all__ = [
    "DenseLayer",
    "NINLayer",
]


class DenseLayer(Layer):
    """
    lasagne.layers.DenseLayer(incoming, num_units,
    W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
    nonlinearity=lasagne.nonlinearities.rectify, num_leading_axes=1, 
    fused_bias=True, **kwargs)

    A fully connected layer.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape

    num_units : int
        The number of units of the layer

    W : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a matrix with shape ``(num_inputs, num_units)``.
        See :func:`lasagne.utils.create_param` for more information.

    b : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_units,)``.
        See :func:`lasagne.utils.create_param` for more information.

    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.

    num_leading_axes : int
        Number of leading axes to distribute the dot product over. These axes
        will be kept in the output tensor, remaining axes will be collapsed and
        multiplied against the weight matrix. A negative number gives the
        (negated) number of trailing axes to involve in the dot product.
    
    fused_bias : bool
        Whether to have a single parameter W for the concatenation of W and b 
        or not.
        
    Examples
    --------
    >>> from lasagne.layers import InputLayer, DenseLayer
    >>> l_in = InputLayer((100, 20))
    >>> l1 = DenseLayer(l_in, num_units=50)

    If the input has more than two axes, by default, all trailing axes will be
    flattened. This is useful when a dense layer follows a convolutional layer.

    >>> l_in = InputLayer((None, 10, 20, 30))
    >>> DenseLayer(l_in, num_units=50).output_shape
    (None, 50)

    Using the `num_leading_axes` argument, you can specify to keep more than
    just the first axis. E.g., to apply the same dot product to each step of a
    batch of time sequences, you would want to keep the first two axes.

    >>> DenseLayer(l_in, num_units=50, num_leading_axes=2).output_shape
    (None, 10, 50)
    >>> DenseLayer(l_in, num_units=50, num_leading_axes=-1).output_shape
    (None, 10, 20, 50)
    """
    def __init__(self, incoming, num_units, W=init.GlorotUniform(),
                 b=init.Constant(0.), nonlinearity=None,
                 num_leading_axes=1, curvature_scale=1,
                 fuse_bias=None, invariant_axes=(2, 3), **kwargs):
        super(DenseLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = nonlinearities.get_nonlinearity(nonlinearity)

        self.num_units = num_units
        self.fuse_bias = get_fuse_bias() if fuse_bias is None else fuse_bias
        self.invariant_axes = invariant_axes
        self.curvature_scale = curvature_scale
        if num_leading_axes >= len(self.input_shape):
            raise ValueError(
                "Got num_leading_axes=%d for a %d-dimensional input, "
                "leaving no trailing axes for the dot product." %
                (num_leading_axes, len(self.input_shape)))
        elif num_leading_axes < -len(self.input_shape):
            raise ValueError(
                "Got num_leading_axes=%d for a %d-dimensional input, "
                "requesting more trailing axes than there are input "
                "dimensions." % (num_leading_axes, len(self.input_shape)))
        self.num_leading_axes = num_leading_axes

        if any(s is None for s in self.input_shape[num_leading_axes:]):
            raise ValueError(
                "A DenseLayer requires a fixed input shape (except for "
                "the leading axes). Got %r for num_leading_axes=%d." %
                (self.input_shape, self.num_leading_axes))
        num_inputs = int(np.prod(self.input_shape[num_leading_axes:]))

        if b is None:
            self.W = self.add_param(W, (num_inputs, num_units), name="W",
                                    broadcast_unit_dims=False)
            self.b = None
            self.W_fused = None
        elif self.fuse_bias:
            self.W_fused = self.add_param(W, (num_inputs + 1, num_units), name="W_fused",
                                          broadcast_unit_dims=False)
            b_val = b.sample((1, num_units))
            w_val = self.W_fused.get_value()
            w_val[-1] = b_val
            self.W_fused.set_value(w_val)
            self.W, self.b = self.params_from_fused(self.W_fused, 2)
        else:
            self.W = self.add_param(W, (num_inputs, num_units), name="W",
                                    broadcast_unit_dims=False)
            self.b = self.add_param(b, (num_units,), name="b",
                                    broadcast_unit_dims=False, regularizable=False)
            self.W_fused = None

    def get_output_shapes_for(self, input_shapes):
        input_shape = input_shapes[0]
        return input_shape[:self.num_leading_axes] + (self.num_units, ),

    def transform_x(self, x):
        num_leading_axes = self.num_leading_axes
        if num_leading_axes < 0:
            num_leading_axes += x.ndim
        if x.ndim > num_leading_axes + 1:
            # flatten trailing axes (into (n+1)-tensor for num_leading_axes=n)
            x = x.flatten(num_leading_axes + 1)
        return x

    def get_outputs_for(self, inputs, **kwargs):
        x = self.transform_x(inputs[0])

        if self.b is None:
            activation = T.dot(x, self.W)
        elif self.fuse_bias:
            x = T.concatenate((x, T.ones((x.shape[0], 1))), axis=1)
            activation = T.dot(x, self.W_fused)
        else:
            activation = T.dot(x, self.W) + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation),

    def params_to_fused(self, params):
        if len(params) == 1:
            return params[0]
        else:
            return T.concatenate((params[0], T.reshape(params[1], (1, -1))), axis=0)

    def params_from_fused(self, fused, num):
        if num == 1:
            return fused,
        else:
            return fused[:-1], fused[-1]

    def curvature_propagation(self, optimizer, inputs, outputs, curvature, make_matrix):
        assert len(inputs) == 1
        assert len(curvature) == 1
        assert len(outputs) == 1

        # Extract the pre-activation of the layer - W * x + b
        bias = self.b is not None and not self.fuse_bias
        activation = fetch_pre_activation(outputs[0], inputs[0], T.basic.Dot, bias)
        # Extract info and calculate Q
        x = self.transform_x(inputs[0])
        curvature = curvature[0]
        q_dim, g_dim = self.W.shape.eval()
        if self.b is not None:
            x = T.concatenate((x, T.ones((x.shape[0], 1))), axis=1)
            q_dim += 1
        if optimizer.variant == "kfra":
            n = th_fx(x.shape[0])
            q = T.dot(x.T, x) / n
            # Propagate curvature trough the nonlinearity and calculate the activation GN
            if self.nonlinearity != nonlinearities.identity:
                a = T.Lop(outputs[0], activation, T.ones_like(outputs[0]))
                g = curvature * T.dot(a.T, a) / n
            else:
                g = curvature
            q *= self.curvature_scale
            g *= self.curvature_scale
            make_matrix(self, g_dim, q_dim, g, q, self.get_params())
            if len(self.input_shapes[0]) > 2:
                if self.invariant_axes != (2, 3):
                    raise ValueError("You are using KFRA on a dense layer with an input of size more than 2D."
                                     "However, this requires that you are invariant on axis 2 and 3 which you are"
                                     "not. Ask yourself - But why?")
                mn = inputs[0].shape[2] * inputs[0].shape[3]
                w = T.reshape(self.W, (inputs[0].shape[1], mn, self.W.shape[1]))
                w = T.sum(w, axis=1)
                return T.dot(T.dot(w, g), w.T),
            else:
                return T.dot(T.dot(self.W, g), self.W.T),
        else:
            # Propagate curvature trough the nonlinearity and calculate the activation GN
            if self.nonlinearity != nonlinearities.identity:
                curvature = T.Lop(outputs[0], activation, curvature)
            x *= T.sqrt(self.curvature_scale)
            curvature *= T.sqrt(self.curvature_scale)
            make_matrix(self, g_dim, q_dim, curvature, x, self.get_params())
            return T.Lop(activation, inputs[0], curvature),


class NINLayer(Layer):
    """
    lasagne.layers.NINLayer(incoming, num_units, untie_biases=False,
    W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
    nonlinearity=lasagne.nonlinearities.rectify, **kwargs)

    Network-in-network layer.
    Like DenseLayer, but broadcasting across all trailing dimensions beyond the
    2nd.  This results in a convolution operation with filter size 1 on all
    trailing dimensions.  Any number of trailing dimensions is supported,
    so NINLayer can be used to implement 1D, 2D, 3D, ... convolutions.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape

    num_units : int
        The number of units of the layer

    untie_biases : bool
        If false the network has a single bias vector similar to a dense
        layer. If true a separate bias vector is used for each trailing
        dimension beyond the 2nd.

    W : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a matrix with shape ``(num_inputs, num_units)``,
        where ``num_inputs`` is the size of the second dimension of the input.
        See :func:`lasagne.utils.create_param` for more information.

    b : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_units,)`` for ``untie_biases=False``, and
        a tensor of shape ``(num_units, input_shape[2], ..., input_shape[-1])``
        for ``untie_biases=True``.
        See :func:`lasagne.utils.create_param` for more information.

    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.

    Examples
    --------
    >>> from lasagne.layers import InputLayer, NINLayer
    >>> l_in = InputLayer((100, 20, 10, 3))
    >>> l1 = NINLayer(l_in, num_units=5)

    References
    ----------
    .. [1] Lin, Min, Qiang Chen, and Shuicheng Yan (2013):
           Network in network. arXiv preprint arXiv:1312.4400.
    """
    def __init__(self, incoming, num_units, untie_biases=False,
                 W=init.GlorotUniform(), b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify, **kwargs):
        super(NINLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

        self.num_units = num_units
        self.untie_biases = untie_biases

        num_input_channels = self.input_shape[1]

        self.W = self.add_param(W, (num_input_channels, num_units), name="W")
        if b is None:
            self.b = None
        else:
            if self.untie_biases:
                biases_shape = (num_units,) + self.output_shape[2:]
            else:
                biases_shape = (num_units,)
            self.b = self.add_param(b, biases_shape, name="b",
                                    regularizable=False)

    def get_output_shapes_for(self, input_shapes):
        input_shape = input_shapes[0]
        return (input_shape[0], self.num_units) + input_shape[2:],

    def get_outputs_for(self, inputs, **kwargs):
        x = inputs[0]
        # cf * bc01... = fb01...
        out_r = T.tensordot(self.W, x, axes=[[0], [1]])
        # input dims to broadcast over
        remaining_dims = range(2, x.ndim)
        # bf01...
        out = out_r.dimshuffle(1, 0, *remaining_dims)

        if self.b is None:
            activation = out
        else:
            if self.untie_biases:
                # no broadcast
                remaining_dims_biases = range(1, x.ndim - 1)
            else:
                remaining_dims_biases = ['x'] * (x.ndim - 2)  # broadcast
            b_shuffled = self.b.dimshuffle('x', 0, *remaining_dims_biases)
            activation = out + b_shuffled

        return self.nonlinearity(activation),
