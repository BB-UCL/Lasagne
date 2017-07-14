# -*- coding: utf-8 -*-

"""
The :class:`LocalResponseNormalization2DLayer
<lasagne.layers.LocalResponseNormalization2DLayer>` implementation contains
code from `pylearn2 <http://github.com/lisa-lab/pylearn2>`_, which is covered
by the following license:


Copyright (c) 2011--2014, Université de Montréal
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import theano
import theano.tensor as T
import numpy as np

from .. import init
from .. import nonlinearities
from .. import utils

from .base import Layer


__all__ = [
    "LocalResponseNormalization2DLayer",
    "BatchNormLayer",
    "batch_norm",
    "BatchNormTheanoLayer",
    "LayerNormLayer",
    "layer_norm"
]


class LocalResponseNormalization2DLayer(Layer):
    """
    Cross-channel Local Response Normalization for 2D feature maps.

    Aggregation is purely across channels, not within channels,
    and performed "pixelwise".

    If the value of the :math:`i` th channel is :math:`x_i`, the output is

    .. math::
        x_i = \\frac{x_i}{ (k + ( \\alpha \\sum_j x_j^2 ))^\\beta }

    where the summation is performed over this position on :math:`n`
    neighboring channels.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape. Must
        follow *BC01* layout, i.e., ``(batchsize, channels, rows, columns)``.
    alpha : float scalar
        coefficient, see equation above
    k : float scalar
        offset, see equation above
    beta : float scalar
        exponent, see equation above
    n : int
        number of adjacent channels to normalize over, must be odd

    Notes
    -----
    This code is adapted from pylearn2. See the module docstring for license
    information.
    """

    def __init__(self, incoming, alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):
        super(LocalResponseNormalization2DLayer, self).__init__(incoming,
                                                                **kwargs)
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n
        if n % 2 == 0:
            raise NotImplementedError("Only works with odd n")

    def get_output_shapes_for(self, input_shapes):
        return input_shapes

    def get_outputs_for(self, inputs, **kwargs):
        x = inputs[0]
        input_shape = self.input_shape
        if any(s is None for s in input_shape):
            input_shape = x.shape
        half_n = self.n // 2
        input_sqr = T.sqr(x)
        b, ch, r, c = input_shape
        extra_channels = T.alloc(0., b, ch + 2*half_n, r, c)
        input_sqr = T.set_subtensor(extra_channels[:, half_n:half_n+ch, :, :],
                                    input_sqr)
        scale = self.k
        for i in range(self.n):
            scale += self.alpha * input_sqr[:, i:i+ch, :, :]
        scale = scale ** self.beta
        return x / scale,


class BatchNormLayer(Layer):
    """
    lasagne.layers.BatchNormLayer(incoming, axes='auto', epsilon=1e-4,
    alpha=0.1, beta=lasagne.init.Constant(0), gamma=lasagne.init.Constant(1),
    mean=lasagne.init.Constant(0), inv_std=lasagne.init.Constant(1), **kwargs)

    Batch Normalization

    This layer implements batch normalization of its inputs, following [1]_:

    .. math::
        y = \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} \\gamma + \\beta

    That is, the input is normalized to zero mean and unit variance, and then
    linearly transformed. The crucial part is that the mean and variance are
    computed across the batch dimension, i.e., over examples, not per example.

    During training, :math:`\\mu` and :math:`\\sigma^2` are defined to be the
    mean and variance of the current input mini-batch :math:`x`, and during
    testing, they are replaced with average statistics over the training
    data. Consequently, this layer has four stored parameters: :math:`\\beta`,
    :math:`\\gamma`, and the averages :math:`\\mu` and :math:`\\sigma^2`
    (nota bene: instead of :math:`\\sigma^2`, the layer actually stores
    :math:`1 / \\sqrt{\\sigma^2 + \\epsilon}`, for compatibility to cuDNN).
    By default, this layer learns the average statistics as exponential moving
    averages computed during training, so it can be plugged into an existing
    network without any changes of the training procedure (see Notes).

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    axes : 'auto', 'per-activation', 'spatial', int or tuple of int
        The axis or axes to normalize over. If ``'auto'`` (the default),
        normalize over all axes except for the second: this will normalize over
        the minibatch dimension for dense layers, and additionally over all
        spatial dimensions for convolutional layers.
    epsilon : scalar
        Small constant :math:`\\epsilon` added to the variance before taking
        the square root and dividing by it, to avoid numerical problems
    alpha : scalar
        Coefficient for the exponential moving average of batch-wise means and
        standard deviations computed during training; the closer to one, the
        more it will depend on the last batches seen
    beta : Theano shared variable, expression, numpy array, callable or None
        Initial value, expression or initializer for :math:`\\beta`. Must match
        the incoming shape, skipping all axes in `axes`. Set to ``None`` to fix
        it to 0.0 instead of learning it.
        See :func:`lasagne.utils.create_param` for more information.
    gamma : Theano shared variable, expression, numpy array, callable or None
        Initial value, expression or initializer for :math:`\\gamma`. Must
        match the incoming shape, skipping all axes in `axes`. Set to ``None``
        to fix it to 1.0 instead of learning it.
        See :func:`lasagne.utils.create_param` for more information.
    mean : Theano shared variable, expression, numpy array, or callable
        Initial value, expression or initializer for :math:`\\mu`. Must match
        the incoming shape, skipping all axes in `axes`.
        See :func:`lasagne.utils.create_param` for more information.
    inv_std : Theano shared variable, expression, numpy array, or callable
        Initial value, expression or initializer for :math:`1 / \\sqrt{
        \\sigma^2 + \\epsilon}`. Must match the incoming shape, skipping all
        axes in `axes`.
        See :func:`lasagne.utils.create_param` for more information.
    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.

    Notes
    -----
    This layer should be inserted between a linear transformation (such as a
    :class:`DenseLayer`, or :class:`Conv2DLayer`) and its nonlinearity. The
    convenience function :func:`batch_norm` modifies an existing layer to
    insert batch normalization in front of its nonlinearity.

    The behavior can be controlled by passing keyword arguments to
    :func:`lasagne.layers.get_output()` when building the output expression
    of any network containing this layer.

    During training, [1]_ normalize each input mini-batch by its statistics
    and update an exponential moving average of the statistics to be used for
    validation. This can be achieved by passing ``deterministic=False``.
    For validation, [1]_ normalize each input mini-batch by the stored
    statistics. This can be achieved by passing ``deterministic=True``.

    For more fine-grained control, ``batch_norm_update_averages`` can be passed
    to update the exponential moving averages (``True``) or not (``False``),
    and ``batch_norm_use_averages`` can be passed to use the exponential moving
    averages for normalization (``True``) or normalize each mini-batch by its
    own statistics (``False``). These settings override ``deterministic``.

    Note that for testing a model after training, [1]_ replace the stored
    exponential moving average statistics by fixing all network weights and
    re-computing average statistics over the training data in a layerwise
    fashion. This is not part of the layer implementation.

    In case you set `axes` to not include the batch dimension (the first axis,
    usually), normalization is done per example, not across examples. This does
    not require any averages, so you can pass ``batch_norm_update_averages``
    and ``batch_norm_use_averages`` as ``False`` in this case.

    See also
    --------
    batch_norm : Convenience function to apply batch normalization to a layer

    References
    ----------
    .. [1] Ioffe, Sergey and Szegedy, Christian (2015):
           Batch Normalization: Accelerating Deep Network Training by Reducing
           Internal Covariate Shift. http://arxiv.org/abs/1502.03167.
    """
    def __init__(self, incoming, axes='auto', epsilon=1e-4, alpha=0.1,
                 beta=init.Constant(0), gamma=init.Constant(1),
                 mean=init.Constant(0), inv_std=init.Constant(1), **kwargs):
        super(BatchNormLayer, self).__init__(incoming, **kwargs)
        if axes in ('auto', 'spatial'):
            # default: normalize over all but the second axis
            axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif axes == 'per-activation':
            axes = (0, )
        elif isinstance(axes, int):
            axes = (axes,)
        self.axes = axes

        self.epsilon = epsilon
        self.alpha = alpha

        # create parameters, ignoring all dimensions in axes
        shape = tuple(size for axis, size in enumerate(self.input_shape)
                      if axis not in self.axes)
        if any(size is None for size in shape):
            raise ValueError("BatchNormLayer needs specified input sizes for "
                             "all axes not normalized over.")
        if beta is None:
            self.beta = None
        else:
            self.beta = self.add_param(beta, shape, 'beta',
                                       trainable=True, regularizable=False)
        if gamma is None:
            self.gamma = None
        else:
            self.gamma = self.add_param(gamma, shape, 'gamma',
                                        trainable=True, regularizable=True)
        self.mean = self.add_param(mean, shape, 'mean',
                                   trainable=False, regularizable=False)
        self.inv_std = self.add_param(inv_std, shape, 'inv_std',
                                      trainable=False, regularizable=False)

    def get_outputs_for(self, inputs, deterministic=False,
                        batch_norm_use_averages=None,
                        batch_norm_update_averages=None, **kwargs):
        x = inputs[0]
        input_mean = x.mean(self.axes)
        input_inv_std = T.inv(T.sqrt(x.var(self.axes) + self.epsilon))

        # Decide whether to use the stored averages or mini-batch statistics
        if batch_norm_use_averages is None:
            batch_norm_use_averages = deterministic
        use_averages = batch_norm_use_averages

        if use_averages:
            mean = self.mean
            inv_std = self.inv_std
        else:
            mean = input_mean
            inv_std = input_inv_std

        # Decide whether to update the stored averages
        if batch_norm_update_averages is None:
            batch_norm_update_averages = not deterministic
        update_averages = batch_norm_update_averages

        if update_averages:
            # Trick: To update the stored statistics, we create memory-aliased
            # clones of the stored statistics:
            running_mean = theano.clone(self.mean, share_inputs=False)
            running_inv_std = theano.clone(self.inv_std, share_inputs=False)
            # set a default update for them:
            running_mean.default_update = ((1 - self.alpha) * running_mean +
                                           self.alpha * input_mean)
            running_inv_std.default_update = ((1 - self.alpha) *
                                              running_inv_std +
                                              self.alpha * input_inv_std)
            # and make sure they end up in the graph without participating in
            # the computation (this way their default_update will be collected
            # and applied, but the computation will be optimized away):
            mean += 0 * running_mean
            inv_std += 0 * running_inv_std

        # prepare dimshuffle pattern inserting broadcastable axes as needed
        param_axes = iter(range(x.ndim - len(self.axes)))
        pattern = ['x' if input_axis in self.axes
                   else next(param_axes)
                   for input_axis in range(x.ndim)]

        # apply dimshuffle pattern to all parameters
        beta = 0 if self.beta is None else self.beta.dimshuffle(pattern)
        gamma = 1 if self.gamma is None else self.gamma.dimshuffle(pattern)
        mean = mean.dimshuffle(pattern)
        inv_std = inv_std.dimshuffle(pattern)

        # normalize
        normalized = (x - mean) * (gamma * inv_std) + beta
        return normalized,

    def params_to_fused(self, params):
        if len(params) == 1:
            return params[0].reshape((-1, 1))
        else:
            return T.concatenate((params[0].reshape((-1, 1)), params[1].reshape((-1, 1))), axis=0)

    def params_from_fused(self, fused, num):
        ndim = len(self.input_shape)
        shape = tuple(self.input_shape[a] for a in range(ndim) if a not in self.axes)
        if num == 1:
            return T.reshape(fused, shape),
        else:
            n = T.int_div(fused.shape[0], 2)
            return T.reshape(fused[:n], shape), T.reshape(fused[n:], shape)

    def curvature_propagation(self, optimizer, inputs, outputs, curvature, make_matrix):
        assert len(inputs) == 1
        assert len(curvature) == 1
        assert len(outputs) == 1
        curvature = curvature[0]

        if optimizer.variant == "kfra":
            raise NotImplementedError
        else:
            ndim = len(self.input_shape)
            non_axes = tuple(i for i in range(ndim) if i not in self.axes)
            shuffle = self.axes + non_axes
            shape = inputs[0].shape
            d1 = T.prod([shape[i] for i in self.axes])
            d2 = np.prod(self.mean.get_value().shape)
            g_dim = 1
            g = T.constant(1.0).reshape((1, 1))
            q_dim = d2
            if self.beta is not None:
                q = curvature.dimshuffle(*shuffle).reshape((d1, d2))
                make_matrix(self, g_dim, q_dim, g, q, (self.beta, ), name="beta")
            if self.gamma is not None:
                if self.beta:
                    centered_plus_beat = outputs[0].owner.inputs
                    assert len(centered_plus_beat) == 2
                    if isinstance(centered_plus_beat[0].owner.op, theano.tensor.elemwise.DimShuffle):
                        centered_times_gamma = centered_plus_beat[1]
                    elif isinstance(centered_plus_beat[1].owner.op, theano.tensor.elemwise.DimShuffle):
                        centered_times_gamma = centered_plus_beat[0]
                    else:
                        raise ValueError("Did not find dimshuffle")
                else:
                    centered_times_gamma = outputs
                op = centered_times_gamma.owner.inputs[0].owner.op
                if isinstance(op, theano.tensor.elemwise.Elemwise):
                    if isinstance(op.scalar_op, theano.scalar.basic.Sub):
                        centered = centered_times_gamma.owner.inputs[0]
                        factor = centered_times_gamma.owner.inputs[1]
                    else:
                        centered = centered_times_gamma.owner.inputs[1]
                        factor = centered_times_gamma.owner.inputs[0]
                else:
                    raise ValueError("First child of centered_times_gamma is not Elemwise")
                assert isinstance(factor.owner.op, theano.tensor.elemwise.Elemwise)
                assert isinstance(factor.owner.op.scalar_op, theano.scalar.basic.Mul)
                if factor.owner.inputs[0] == self.gamma:
                    inv_std = factor.owner.inputs[1]
                elif factor.owner.inputs[0].owner.inputs[0] == self.gamma:
                    inv_std = factor.owner.inputs[1]
                else:
                    inv_std = factor.owner.inputs[0]
                normalized = centered * inv_std
                q1 = curvature.dimshuffle(*shuffle).reshape((d1, d2))
                q2 = normalized.dimshuffle(*shuffle).reshape((d1, d2))
                q = q1 * q2
                make_matrix(self, g_dim, q_dim, g, q, (self.gamma,), name="gamma")
            return T.Lop(outputs[0], inputs[0], curvature),


class BatchNormTheanoLayer(Layer):
    """
    lasagne.layers.BatchNormLayer(incoming, axes='auto', epsilon=1e-4,
    alpha=0.1, beta=lasagne.init.Constant(0), gamma=lasagne.init.Constant(1),
    mean=lasagne.init.Constant(0), inv_std=lasagne.init.Constant(1), **kwargs)

    Batch Normalization

    This layer implements batch normalization of its inputs, following [1]_:

    .. math::
        y = \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} \\gamma + \\beta

    That is, the input is normalized to zero mean and unit variance, and then
    linearly transformed. The crucial part is that the mean and variance are
    computed across the batch dimension, i.e., over examples, not per example.

    During training, :math:`\\mu` and :math:`\\sigma^2` are defined to be the
    mean and variance of the current input mini-batch :math:`x`, and during
    testing, they are replaced with average statistics over the training
    data. Consequently, this layer has four stored parameters: :math:`\\beta`,
    :math:`\\gamma`, and the averages :math:`\\mu` and :math:`\\sigma^2`
    (nota bene: instead of :math:`\\sigma^2`, the layer actually stores
    :math:`1 / \\sqrt{\\sigma^2 + \\epsilon}`, for compatibility to cuDNN).
    By default, this layer learns the average statistics as exponential moving
    averages computed during training, so it can be plugged into an existing
    network without any changes of the training procedure (see Notes).

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    axes : 'auto', 'per-activation', 'spatial', int or tuple of int
        The axis or axes to normalize over. If ``'auto'`` (the default),
        normalize over all axes except for the second: this will normalize over
        the minibatch dimension for dense layers, and additionally over all
        spatial dimensions for convolutional layers.
    epsilon : scalar
        Small constant :math:`\\epsilon` added to the variance before taking
        the square root and dividing by it, to avoid numerical problems
    alpha : scalar
        Coefficient for the exponential moving average of batch-wise means and
        standard deviations computed during training; the closer to one, the
        more it will depend on the last batches seen
    beta : Theano shared variable, expression, numpy array, callable or None
        Initial value, expression or initializer for :math:`\\beta`. Must match
        the incoming shape, skipping all axes in `axes`. Set to ``None`` to fix
        it to 0.0 instead of learning it.
        See :func:`lasagne.utils.create_param` for more information.
    gamma : Theano shared variable, expression, numpy array, callable or None
        Initial value, expression or initializer for :math:`\\gamma`. Must
        match the incoming shape, skipping all axes in `axes`. Set to ``None``
        to fix it to 1.0 instead of learning it.
        See :func:`lasagne.utils.create_param` for more information.
    mean : Theano shared variable, expression, numpy array, or callable
        Initial value, expression or initializer for :math:`\\mu`. Must match
        the incoming shape, skipping all axes in `axes`.
        See :func:`lasagne.utils.create_param` for more information.
    inv_std : Theano shared variable, expression, numpy array, or callable
        Initial value, expression or initializer for :math:`1 / \\sqrt{
        \\sigma^2 + \\epsilon}`. Must match the incoming shape, skipping all
        axes in `axes`.
        See :func:`lasagne.utils.create_param` for more information.
    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.

    Notes
    -----
    This layer should be inserted between a linear transformation (such as a
    :class:`DenseLayer`, or :class:`Conv2DLayer`) and its nonlinearity. The
    convenience function :func:`batch_norm` modifies an existing layer to
    insert batch normalization in front of its nonlinearity.

    The behavior can be controlled by passing keyword arguments to
    :func:`lasagne.layers.get_output()` when building the output expression
    of any network containing this layer.

    During training, [1]_ normalize each input mini-batch by its statistics
    and update an exponential moving average of the statistics to be used for
    validation. This can be achieved by passing ``deterministic=False``.
    For validation, [1]_ normalize each input mini-batch by the stored
    statistics. This can be achieved by passing ``deterministic=True``.

    For more fine-grained control, ``batch_norm_update_averages`` can be passed
    to update the exponential moving averages (``True``) or not (``False``),
    and ``batch_norm_use_averages`` can be passed to use the exponential moving
    averages for normalization (``True``) or normalize each mini-batch by its
    own statistics (``False``). These settings override ``deterministic``.

    Note that for testing a model after training, [1]_ replace the stored
    exponential moving average statistics by fixing all network weights and
    re-computing average statistics over the training data in a layerwise
    fashion. This is not part of the layer implementation.

    In case you set `axes` to not include the batch dimension (the first axis,
    usually), normalization is done per example, not across examples. This does
    not require any averages, so you can pass ``batch_norm_update_averages``
    and ``batch_norm_use_averages`` as ``False`` in this case.

    See also
    --------
    batch_norm : Convenience function to apply batch normalization to a layer

    References
    ----------
    .. [1] Ioffe, Sergey and Szegedy, Christian (2015):
           Batch Normalization: Accelerating Deep Network Training by Reducing
           Internal Covariate Shift. http://arxiv.org/abs/1502.03167.
    """
    def __init__(self, incoming, axes='auto', epsilon=1e-4, alpha=0.1,
                 beta=init.Constant(0), gamma=init.Constant(1),
                 mean=init.Constant(0), inv_std=init.Constant(1), **kwargs):
        super(BatchNormTheanoLayer, self).__init__(incoming, **kwargs)
        ndim = len(self.input_shape)
        if axes in ('auto', 'spatial'):
            axes = (0,) + tuple(range(2, ndim))
        elif axes == 'per-activation':
            axes = (0,)
        elif isinstance(axes, (tuple, list, np.ndarray)):
            axes = tuple(int(a) for a in axes)
        else:
            raise ValueError('invalid axes: %s', str(axes))
        axes = tuple(sorted(axes))
        if len(axes) == 0:
            raise ValueError('there should be at least one normalization axis')
        if min(axes) < 0 or max(axes) >= ndim:
            raise ValueError('axes should be less than ndim (<%d), but %s given' % (ndim, str(axes)))
        self.axes = axes
        self.epsilon = epsilon
        self.alpha = alpha

        # create parameters, ignoring all dimensions in axes
        shape = utils.get_shape_except_axes(self.input_shape, axes)
        if beta is None:
            self.beta = None
        else:
            self.beta = self.add_param(beta, shape, 'beta',
                                       trainable=True, regularizable=False)
        if gamma is None:
            self.gamma = None
        else:
            self.gamma = self.add_param(gamma, shape, 'gamma',
                                        trainable=True, regularizable=True)
        self.mean = self.add_param(mean, shape, 'mean',
                                   trainable=False, regularizable=False)
        self.var = self.add_param(inv_std, shape, 'var',
                                  trainable=False, regularizable=False)

    def get_outputs_for(self, inputs, deterministic=False,
                        batch_norm_use_averages=None,
                        batch_norm_update_averages=None, **kwargs):
        x = inputs[0]

        # Decide whether to use the stored averages or mini-batch statistics
        if batch_norm_use_averages is None:
            batch_norm_use_averages = deterministic
        use_averages = batch_norm_use_averages

        # Decide whether to update the stored averages
        if batch_norm_update_averages is None:
            batch_norm_update_averages = not deterministic
        update_averages = batch_norm_update_averages

        gamma = 1 if self.gamma is None else self.gamma
        beta = 0 if self.beta is None else self.beta
        if use_averages:
            bn = T.nnet.bn.batch_normalization_test
            out = bn(x, gamma, beta,
                     self.mean, self.var,
                     self.axes, self.epsilon)
        elif update_averages:
            bn = T.nnet.bn.batch_normalization_train
            # Trick: To update the stored statistics, we create memory-aliased
            # clones of the stored statistics:
            out, _, _, new_mean, new_var = bn(x, gamma, beta, self.axes,
                                              self.epsilon, self.alpha,
                                              self.mean, self.var)
            # set a default update for them:
            self.mean.default_update = new_mean
            self.var.default_update = new_var
            # and make sure they end up in the graph without participating in
            # the computation (this way their default_update will be collected
            # and applied, but the computation will be optimized away):
            pattern = ['x' for _ in range(out.ndim)]
            c = 0
            for i in range(out.ndim):
                if i not in self.axes:
                    pattern[i] = c
                    c += 1
            out += 0 * self.mean.dimshuffle(*pattern)
            out += 0 * self.var.dimshuffle(*pattern)
        else:
            bn = T.nnet.bn.batch_normalization_train
            out, _, _, _, _ = bn(x, gamma, beta, self.axes,
                                 self.epsilon, self.alpha)

        return out,

    def params_to_fused(self, params):
        if len(params) == 1:
            return params[0].reshape((-1, 1))
        else:
            return T.concatenate((params[0].reshape((-1, 1)), params[1].reshape((-1, 1))), axis=0)

    def params_from_fused(self, fused, num):
        ndim = len(self.input_shape)
        shape = tuple(self.input_shape[a] for a in range(ndim) if a not in self.axes)
        if num == 1:
            return T.reshape(fused, shape),
        else:
            n = T.int_div(fused.shape[0], 2)
            return T.reshape(fused[:n], shape), T.reshape(fused[n:], shape)

    def curvature_propagation(self, optimizer, inputs, outputs, curvature, make_matrix):
        assert len(inputs) == 1
        assert len(curvature) == 1
        assert len(outputs) == 1
        curvature = curvature[0]

        if optimizer.variant == "kfra":
            raise NotImplementedError
        else:
            ndim = len(self.input_shape)
            non_axes = tuple(i for i in range(ndim) if i not in self.axes)
            shuffle = self.axes + non_axes
            shape = inputs[0].shape
            d1 = T.prod([shape[i] for i in self.axes])
            d2 = np.prod(self.mean.get_value().shape)
            g_dim = 1
            g = T.constant(1.0).reshape((1, 1))
            q_dim = d2
            if self.beta is not None:
                q = curvature.dimshuffle(*shuffle).reshape((d1, d2))
                make_matrix(self, g_dim, q_dim, g, q, (self.beta, ), name="beta")
            if self.gamma is not None:
                bn_op = utils.fetch_pre_activation(outputs[0], self.beta, T.nnet.bn.AbstractBatchNormTrain).owner
                mean, inv_std = bn_op.outputs[1], bn_op.outputs[2]
                normalized = (inputs[0] - mean) * inv_std
                q1 = curvature.dimshuffle(*shuffle).reshape((d1, d2))
                q2 = normalized.dimshuffle(*shuffle).reshape((d1, d2))
                q = q1 * q2
                make_matrix(self, g_dim, q_dim, g, q, (self.gamma,), name="gamma")
            return T.Lop(outputs[0], inputs[0], curvature),


def batch_norm(layer, use_theano=True, **kwargs):
    """
    Apply batch normalization to an existing layer. This is a convenience
    function modifying an existing layer to include batch normalization: It
    will steal the layer's nonlinearity if there is one (effectively
    introducing the normalization right before the nonlinearity), remove
    the layer's bias if there is one (because it would be redundant), and add
    a :class:`BatchNormLayer` and :class:`NonlinearityLayer` on top.

    Parameters
    ----------
    layer : A :class:`Layer` instance
        The layer to apply the normalization to; note that it will be
        irreversibly modified as specified above
    use_theano : bool (default: True)
        Whether to se :class: `BatchNormTheanoLayer` or
        :class: `BatchNormLayer`.
    **kwargs
        Any additional keyword arguments are passed on to the
        :class:`BatchNormLayer` constructor.

    Returns
    -------
    BatchNormLayer or NonlinearityLayer instance
        A batch normalization layer stacked on the given modified `layer`, or
        a nonlinearity layer stacked on top of both if `layer` was nonlinear.

    Examples
    --------
    Just wrap any layer into a :func:`batch_norm` call on creating it:

    >>> from lasagne.layers import InputLayer, DenseLayer, batch_norm
    >>> from lasagne.nonlinearities import tanh
    >>> l1 = InputLayer((64, 768))
    >>> l2 = batch_norm(DenseLayer(l1, num_units=500, nonlinearity=tanh))

    This introduces batch normalization right before its nonlinearity:

    >>> from lasagne.layers import get_all_layers
    >>> [l.__class__.__name__ for l in get_all_layers(l2)]
    ['InputLayer', 'DenseLayer', 'BatchNormLayer', 'NonlinearityLayer']
    """
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = nonlinearities.identity
    if hasattr(layer, 'b') and layer.b is not None:
        del layer.params[layer.b]
        layer.b = None
    bn_name = (kwargs.pop('name', None) or
               (getattr(layer, 'name', None) and layer.name + '_bn'))
    if use_theano:
        layer = BatchNormTheanoLayer(layer, name=bn_name, **kwargs)
    else:
        layer = BatchNormLayer(layer, name=bn_name, **kwargs)
    if nonlinearity is not None:
        from .special import NonlinearityLayer
        nonlin_name = bn_name and bn_name + '_nonlin'
        layer = NonlinearityLayer(layer, nonlinearity, name=nonlin_name)
    return layer


class LayerNormLayer(Layer):
    def __init__(self, incoming, axes='auto', epsilon=1e-4,
                 beta=init.Constant(0), gamma=init.Constant(1),
                 **kwargs):
        super(LayerNormLayer, self).__init__(incoming, **kwargs)
        self.axes = utils.get_layer_norm_axes(axes, len(self.input_shape))
        self.epsilon = epsilon

        # create parameters, ignoring all dimensions in axes
        shape = tuple(self.input_shape[a] for a in range(len(self.input_shape)) if a in self.axes)
        if any(size is None for size in shape):
            raise ValueError("LayerNormLayer needs specified input sizes for "
                             "all axes not normalized over.")
        if beta is None:
            self.beta = None
        else:
            self.beta = self.add_param(beta, shape, 'beta',
                                       trainable=True, regularizable=False)
        if gamma is None:
            self.gamma = None
        else:
            self.gamma = self.add_param(gamma, shape, 'gamma',
                                        trainable=True, regularizable=True)

    def get_outputs_for(self, inputs, **kwargs):
        x = inputs[0]
        return utils.layer_norm_tensor(x, self.gamma, self.beta, self.axes, self.epsilon),

    def params_to_fused(self, params):
        if len(params) == 1:
            return params[0].reshape((-1, 1))
        else:
            return T.concatenate((params[0].reshape((-1, 1)), params[1].reshape((-1, 1))), axis=0)

    def params_from_fused(self, fused, num):
        ndim = len(self.input_shape)
        shape = tuple(self.input_shape[a] for a in range(ndim) if a not in self.axes)
        if num == 1:
            return T.reshape(fused, shape),
        else:
            n = T.int_div(fused.shape[0], 2)
            return T.reshape(fused[:n], shape), T.reshape(fused[n:], shape)

    def curvature_propagation(self, optimizer, inputs, outputs, curvature, make_matrix):
        raise NotImplementedError
        assert len(inputs) == 1
        assert len(curvature) == 1
        assert len(outputs) == 1
        curvature = curvature[0]

        if optimizer.variant == "kfra":
            raise NotImplementedError
        else:
            ndim = len(self.input_shape)
            non_axes = tuple(i for i in range(ndim) if i not in self.axes)
            shuffle = self.axes + non_axes
            shape = inputs[0].shape
            d1 = T.prod([shape[i] for i in self.axes])
            d2 = np.prod(self.mean.get_value().shape)
            g_dim = 1
            g = T.constant(1.0).reshape((1, 1))
            q_dim = d2
            if self.beta is not None:
                q = curvature.dimshuffle(*shuffle).reshape((d1, d2))
                make_matrix(self, g_dim, q_dim, g, q, (self.beta, ), name="beta")
            if self.gamma is not None:
                bn_op = fetch_pre_activation(outputs[0], self.beta, T.nnet.bn.AbstractBatchNormTrain).owner
                mean, inv_std = bn_op.outputs[1], bn_op.outputs[2]
                normalized = (inputs[0] - mean) * inv_std
                q1 = curvature.dimshuffle(*shuffle).reshape((d1, d2))
                q2 = normalized.dimshuffle(*shuffle).reshape((d1, d2))
                q = q1 * q2
                make_matrix(self, g_dim, q_dim, g, q, (self.gamma,), name="gamma")
            return T.Lop(outputs[0], inputs[0], curvature),


def layer_norm(layer, **kwargs):
    """
    Apply layer normalization to an existing layer. This is a convenience
    function modifying an existing layer to include layer normalization: It
    will steal the layer's nonlinearity if there is one (effectively
    introducing the normalization right before the nonlinearity), remove
    the layer's bias if there is one (because it would be redundant), and add
    a :class:`LayerNormLayer` and :class:`NonlinearityLayer` on top.

    Parameters
    ----------
    layer : A :class:`Layer` instance
        The layer to apply the normalization to; note that it will be
        irreversibly modified as specified above
    **kwargs
        Any additional keyword arguments are passed on to the
        :class:`LayerNormLayer` constructor.

    Returns
    -------
    LayerNormLayer or NonlinearityLayer instance
        A layer normalization layer stacked on the given modified `layer`, or
        a nonlinearity layer stacked on top of both if `layer` was nonlinear.

    Examples
    --------
    Just wrap any layer into a :func:`layer_norm` call on creating it:

    >>> from lasagne.layers import InputLayer, DenseLayer, layer_norm
    >>> from lasagne.nonlinearities import tanh
    >>> l1 = InputLayer((64, 768))
    >>> l2 = layer_norm(DenseLayer(l1, num_units=500, nonlinearity=tanh))

    This introduces layer normalization right before its nonlinearity:

    >>> from lasagne.layers import get_all_layers
    >>> [l.__class__.__name__ for l in get_all_layers(l2)]
    ['InputLayer', 'DenseLayer', 'LayerNormLayer', 'NonlinearityLayer']
    """
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = nonlinearities.identity
    if hasattr(layer, 'b') and layer.b is not None:
        del layer.params[layer.b]
        layer.b = None
    name = (kwargs.pop('name', None) or
            (getattr(layer, 'name', None) and layer.name + '_ln'))
    layer = LayerNormLayer(layer, name=name, **kwargs)
    if nonlinearity is not None:
        from .special import NonlinearityLayer
        nonlin_name = name and name + '_nonlin'
        layer = NonlinearityLayer(layer, nonlinearity, name=nonlin_name)
    return layer
