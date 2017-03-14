# -*- coding: utf-8 -*-
"""
Layers to construct recurrent networks. Recurrent layers are constructed
on the basis of applying a recurrence function in a loop. The function
is represented via special AbstractStepLayer, which is disconnected with
the outer layers. The RecurrenceLayer owns the step layer as internal
and applies it using a scan function. The inputs can either be in the
format ``(sequence_length, batch_size, num_inputs)``, referred to as
TND ordering, or  ``(batch_size, sequence_length, num_inputs)``
referred to as NTD ordering. In all occasions TND is the default one.

The following recurrent layers are implemented:

.. currentmodule:: lasagne.layers

.. autosummary::
    :nosignatures:

    RecurrenceLayer,
    AbstractStepLayer,
    RNNLayer,
    StandardStep,
    GRUStep,
    LSTMStep,
    RWAStep


Recurrent layers and feed-forward layers can be combined in the same network
by using a few reshape operations; please refer to the example below.

Examples
--------
The following example demonstrates how recurrent layers can be easily mixed
with feed-forward layers using :class:`ReshapeLayer` and how to build a
network with variable batch size and number of time steps.

>>> from lasagne.layers import *
>>> num_inputs, num_units, num_classes = 10, 12, 5
>>> # By setting the first two dimensions as None, we are allowing them to vary
>>> # They correspond to batch size and sequence length, so we will be able to
>>> # feed in batches of varying size with sequences of varying length.
>>> l_inp = InputLayer((None, None, num_inputs))
>>> # We can retrieve symbolic references to the input variable's shape, which
>>> # we will later use in reshape layers.
>>> batchsize, seqlen, _ = l_inp.input_var.shape
>>> lstm_step = LSTMStep((None, num_inputs), num_units)
>>> l_rec = RNNLayer(l_inp, lstm_step)
>>> # In order to connect a recurrent layer to a dense layer, we need to
>>> # flatten the first two dimensions (our "sample dimensions"); this will
>>> # cause each time step of each sequence to be processed independently
>>> l_shp = ReshapeLayer(l_rec, (-1, num_units))
>>> l_dense = DenseLayer(l_shp, num_units=num_classes)
>>> # To reshape back to our original shape, we can use the symbolic shape
>>> # variables we retrieved above.
>>> l_out = ReshapeLayer(l_dense, (batchsize, seqlen, num_classes))
"""
import theano
import theano.tensor as T
from .. import nonlinearities
from .. import init
from .. import utils

from .base import Layer, IndexLayer
from .dense import DenseLayer
from .merge import ConcatLayer
from .input import InputLayer
from . import helper

__all__ = [
    "RecurrenceLayer",
    "AbstractStepLayer",
    "RNNLayer",
    "StandardStep",
    "GRUStep",
    "LSTMStep",
    "RWAStep"
]


class RecurrenceLayer(Layer):
    """
    lasagne.layers.recurrent.RecurrenceLayer(incoming, step_layer,
    in_to_hid=None, hid_to_out=None, mask_input=None, pre_compute_input=True,
    in_order="TND", out_order="TND", backwards=False, gradient_steps=-1,
    only_return_final=False, unroll_scan=False, **kwargs)

    A layer which applies the function defined by the step_layer in a loop
    over the inputs.

    This layer allows you to specify custom input-to-hidden and
    hidden-to-output layers by instantiating :class:`lasagne.layers.Layer`
    instances and passing them on initialization.  Note that these connections
    can consist of multiple layers chained together.  The output shape for the
    provided input-to-hidden and hidden-to-hidden connections must be the same.
    If you are looking for a standard recurrent layer please see
    :class:`RNNLayer` and the corresponding step layers.

    Parameters
    ----------
    incoming : a tuple of either :class:`lasagne.layers.Layer`
        or tuples specifying the inputs shapes feeding into this layer.

    step_layer : :class:`lasagne.layers.AbstractStepLayer`
        A layer which defines the function that would be applied on each step
         of the loop. The layer itself can be a chain of other layers.

    in_to_hid : :class:`lasagne.layers.Layer`
        :class:`lasagne.layers.Layer` instance which connects inputs to the
        hidden state (:math:`f_i`).  This layer may be connected to a chain of
        layers, which must end in with the same number of
        :class:`lasagne.layers.InputLayer` as the number of overall inputs.

    hid_to_out : :class:`lasagne.layers.Layer`
        :class:`lasagne.layers.Layer` instance which processes the outputs
        of the scan before they are finally returned by this layer. This is
        often instantiated as a :class:`lasagne.layers.IndexLayer` for
        recurrent networks with several states passed across the step_layer
        in order to specify which one is used as an output.

    mask_input : :class:`lasagne.layers.Layer`
        Layer which allows for a sequence mask to be input, for when sequences
        are of variable length.  Default `None`, which means no mask will be
        supplied (i.e. all sequences are of the same length).

    precompute_input : bool
        If True, precompute input_to_hid before iterating through
        the sequence. This can result in a speedup at the expense of
        an increase in memory usage.

    in_order : "TND" or "NTD"
        Defines what is the ordering of the inputs. Note that if there are
        several inputs all must be in the same order.

    out_order : "TND" or "NTD"
        Defines what is the ordering of the outputs.

    backwards : bool
        If True, process the sequence backwards and then reverse the
        output again such that the output from the layer is always
        from :math:`x_1` to :math:`x_n`.

    gradient_steps : int
        Number of timesteps to include in the backpropagated gradient.
        If -1, backpropagate through the entire sequence.

    only_return_final : bool
        If True, only return the final sequential output (e.g. for tasks where
        a single target value for the entire sequence is desired).  In this
        case, Theano makes an optimization which saves memory.

    unroll_scan : bool
        If True the recursion is unrolled instead of using scan. For some
        graphs this gives a significant speed up but it might also consume
        more memory. When `unroll_scan` is True, backpropagation always
        includes the full sequence, so `gradient_steps` must be set to -1 and
        the input sequence length must be known at compile time (i.e., cannot
        be given as None).

    In most cases you will not use this layer directly, but via
    :class:`lasagne.layers.RNNLayer`.
    """
    def __init__(self,
                 incoming,
                 step_layer,
                 in_to_hid=None,
                 hid_to_out=None,
                 mask_input=None,
                 pre_compute_input=True,
                 in_order="TND",
                 out_order="TND",
                 backwards=False,
                 gradient_steps=-1,
                 only_return_final=False,
                 unroll_scan=False,
                 **kwargs):
        if isinstance(incoming, (list, tuple)):
            if all(isinstance(i, int) or i is None for i in incoming):
                # Single shape
                incoming = (incoming,)
        else:
            # Single layer
            incoming = (incoming,)
        if mask_input is not None:
            incoming += (mask_input, )
        inner = {"step": step_layer}
        if in_to_hid is not None:
            inner["in_to_hid"] = in_to_hid
        if hid_to_out is not None:
            inner["hid_to_out"] = hid_to_out
        self.mask = mask_input is not None
        self.pre_compute_input = pre_compute_input
        self.in_order = in_order
        self.out_order = out_order
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.only_return_final = only_return_final
        self.unroll_scan = unroll_scan

        # Verification
        if in_order not in ("TND", "NTD"):
            raise ValueError("Unrecognized order:", in_order)
        if out_order not in ("TND", "NTD"):
            raise ValueError("Unrecognized order:", out_order)
        if self.pre_compute_input and in_to_hid is None:
            raise ValueError("If pre_compute_input is True"
                             "you must provide in_to_hid layer")
        # Verify shapes are correct
        super(RecurrenceLayer, self).__init__(incoming,
                                              max_inputs=100,
                                              inner_layers=inner, **kwargs)

    def get_output_shapes_for(self, input_shapes):
        if self.mask:
            input_shapes = input_shapes[:-1]
        shapes = self.get_loop_shapes(input_shapes)
        if self.inner_layers.get("in_to_hid") is not None:
            shapes = self.inner_layers["in_to_hid"] \
                .get_output_shapes_for(shapes)
        shapes = self.inner_layers["step"].get_output_shapes_for(shapes)
        if self.inner_layers.get("hid_to_out") is not None:
            shapes = self.inner_layers["hid_to_out"] \
                .get_output_shapes_for(shapes)
        shapes = self.to_loop_shapes(shapes)
        if self.out_order == "NTD":
            shapes = tuple((s[1], s[0]) + s[2:] for s in shapes)
        return shapes

    def get_outputs_for(self, inputs, **kwargs):
        if self.in_order == "NTD":
            inputs = tuple(i.dimshuffle(*((1, 0) + tuple(range(2, i.ndim))))
                           for i in inputs)

        if self.pre_compute_input:
            layer = self.inner_layers["in_to_hid"]
            t_ns = tuple((i.shape[0], i.shape[1]) for i in inputs)
            inputs = (T.reshape(i, (t1 * t2, -1))
                      for i, (t1, t2) in zip(inputs, t_ns))
            input_layers = (l for l in helper.get_all_layers(layer)
                            if isinstance(l, InputLayer))
            inputs = helper.get_outputs(layer,  dict(zip(input_layers,
                                                         inputs)))
            inputs = tuple(T.reshape(i, (t1, t2, -1))
                           for i, (t1, t2) in zip(inputs, t_ns))

        ns = tuple(i.shape[1] for i in inputs)
        step_l = self.inner_layers["step"]

        def step(*args):
            if self.pre_compute_input:
                return step_l.get_outputs_for(args, **kwargs)
            else:
                n = len(self.input_shapes)
                layer = self.inner_layers["in_to_hid"]
                input_layers = (l for l in helper.get_all_layers(layer)
                                if isinstance(l, InputLayer))
                inputs = helper.get_outputs(layer, dict(zip(input_layers,
                                                            args[:n])))
                return step_l.get_outputs_for(inputs + args[n:], **kwargs)

        if self.unroll_scan:
            if self.gradient_steps != -1:
                raise ValueError("unroll_scan does not support "
                                 "truncating gradient")
            if self.in_order == "TND":
                n_steps = self.input_shapes[0][0]
            else:
                n_steps = self.input_shapes[0][1]
            outputs = utils.unroll_scan(
                fn=step,
                sequences=inputs,
                outputs_info=step_l.get_inits(ns),
                non_sequences=self.get_params(),
                go_backwards=self.backwards,
                n_steps=n_steps)
        else:
            outputs, _ = theano.scan(
                fn=step,
                sequences=inputs,
                outputs_info=step_l.get_inits(ns),
                non_sequences=self.get_params(),
                go_backwards=self.backwards,
                truncate_gradient=self.gradient_steps,
                return_list=True,
                strict=True)

        # If backwards reverse and if only_return_final index
        if self.backwards:
            if self.only_return_final:
                outputs = (o[0] for o in outputs)
            else:
                outputs = (o[::-1] for o in outputs)
        elif self.only_return_final:
            outputs = (o[-1] for o in outputs)

        # Correct order
        if self.out_order == "NTD" and not self.only_return_final:
            outputs = tuple(o.dimshuffle(*((1, 0) + tuple(range(2, o.ndim))))
                            for o in outputs)
        # Apply the hid_to_out layer if exists
        layer = self.inner_layers.get("hid_to_out", None)
        if layer:
            output_layers = (l for l in helper.get_all_layers(layer)
                             if isinstance(l, InputLayer))
            outputs = helper.get_outputs(layer, dict(zip(output_layers,
                                                         outputs)))
        return outputs

    def get_loop_shapes(self, shapes, order=None):
        order = self.in_order if order is None else order
        shapes = utils.shape_to_tuple(shapes)
        if order == "TND":
            return tuple(s[1:] for s in shapes)
        elif order == "NTD":
            return tuple((s[0],) + s[2:] for s in shapes)
        else:
            raise ValueError("Unrecognized order", order)

    def get_loop_shape(self, shape, order=None):
        shape = self.get_loop_shapes(shape, order)
        if len(shape) == 1:
            return shape[0]
        else:
            raise ValueError("More than one shape give, use `get_loop_shapes`")

    def to_loop_shapes(self, shapes, order=None, input_shapes=None):
        order = self.in_order if order is None else order
        input_shapes = self.input_shapes if input_shapes is None \
            else input_shapes
        shapes = utils.shape_to_tuple(shapes)
        t = input_shapes[0][0] if order == "TND" else input_shapes[0][1]
        return tuple((t, ) + s for s in shapes)

    def to_loop_shape(self, shape, order=None, input_shapes=None):
        shape = self.to_loop_shapes(shape, order, input_shapes)
        if len(shape) == 1:
            return shape[0]
        else:
            raise ValueError("More than one shape give, use `to_loop_shapes`")


class RNNLayer(RecurrenceLayer):
    """
    lasagne.layers.recurrent.RNNLayer(incoming, step_layer,
    in_order="TND", out_order="TND", post_indexes=None,
    W=init.Orthogonal(), b=init.Constant(), **kwargs)

    A layer for building standard RNNs. The `step_layer` defines
    the function that will be applied in the loop. For multiple inputs,
    the concatenate them along the third axis. For those cases it
    will construct an `in_to_hid` as :class:`lasagne.layers.ConcatLayer`.
    If pre_compute_input is `True` or step_layer.combine_h_x is False
    it will also add after the concatenation a
    :class:`lasagne.layers.DenseLayer` with no bias and no nonlinearity.
    If the `post_indexes` are not None will construct an `hid_to_out`
    as :class:`lasagne.layers.IndexLayer` which to filter the outputs of
    the step_layer accordingly.

    Parameters
    ----------
    incoming : a tuple of either :class:`lasagne.layers.Layer`
        or tuples specifying the inputs shapes feeding into this layer.

    step_layer : :class:`lasagne.layers.AbstractStepLayer`
        A layer which defines the function that would be applied on each step
        of the loop. The layer itself can be a chain of other layers.

    in_order : "TND" or "NTD"
        Defines what is the ordering of the inputs. Note that if there are
        several inputs all must be in the same order.

    out_order : "TND" or "NTD"
        Defines what is the ordering of the outputs.

    post_indexes : None or tuple of ints
        If it is a tuple those indexes are used to filter the outputs of
        the step_layer. If None the step_layer will be interrogated for
        post_indexes as well.

    W : Theano shared variable, numpy array or callable
        Initializer for the weight matrix of `in_to_hid` (:math:`W_x`)
        if the layer exists (see the description above).

    b : Theano shared variable, numpy array or callable
        Initializer for the weight matrix of `in_to_hid` (:math:`b`)
        if the layer exists (see the description above).

    kwargs : dictionary
        Any extra parameters passed to :class:`lasagne.layers.RecurrenceLayer`
    """
    def __init__(self,
                 incoming,
                 step_layer,
                 in_order="TND",
                 out_order="TND",
                 post_indexes=None,
                 W=init.Orthogonal(),
                 b=init.Constant(),
                 **kwargs):
        in_shapes = helper.get_output_shapes(incoming)
        shapes = self.get_loop_shapes(in_shapes, in_order)
        if len(shapes) > 1:
            # Concatenate all inputs
            inputs = tuple(InputLayer(s) for s in shapes)
            l_in = ConcatLayer(inputs, axis=1, name="in_concat")
        else:
            # Single input
            l_in = InputLayer(shapes[0])
        if step_layer.pre_compute_input or not step_layer.combine_h_x:
            # Dense Wx layer
            l_in = DenseLayer(incoming=l_in,
                              W=W, b=None if step_layer.no_bias else b,
                              num_units=step_layer.num_x_to_h,
                              nonlinearity=nonlinearities.identity,
                              name="in_to_hid")
        post_indexes = post_indexes or step_layer.post_indexes
        l_out = None
        if post_indexes:
            # Post filtering of the step output
            shapes = helper.get_output_shapes(step_layer)
            shapes = self.to_loop_shapes(shapes, in_order, in_shapes)
            inputs = tuple(InputLayer(s) for s in shapes)
            l_out = IndexLayer(inputs, post_indexes)

        super(RNNLayer, self) \
            .__init__(incoming, step_layer,
                      in_to_hid=l_in, hid_to_out=l_out,
                      in_order=in_order, out_order=out_order,
                      pre_compute_input=step_layer.pre_compute_input,
                      **kwargs)


class AbstractStepLayer(Layer):
    """
    lasagne.layers.recurrent.AbstractStepLayer(incoming, num_units,
    pre_compute_input=True, combine_h_x=False, learn_init=True,
    grad_clipping=0, post_filter=None, **kwargs):

    A layer for abstracting only the step function in recurrent layers.
    Some of the arguments are used for the corresponding values in the
    :class:`lasagne.layers.RecurrenceLayer`

    Parameters
    ----------
    incoming : a tuple of either :class:`lasagne.layers.Layer`
        or tuples specifying the inputs shapes feeding into this layer.

    num_x_to_h : int
        Specified the number of units for the transfer of `x`->`h`.
        For standard RNN this is the dimensionality of `h`.
        However, for a GRU for instance this is 3 times more.

    pre_compute_input : bool
        Whether should pre compute the inputs. Also used by the
        :class:`lasagne.layers.RecurrenceLayer`

    combine_h_x : bool
        Whether when we don't precompute the input if we can
        concatenate `x` and `h` thus making the `in_to_hid`
        layer redundant. This is useful in order to have a
        single weight matrix W and thus only one GEMM.

    learn_init : bool
        Whether we should take gradients with respect to the
        initializations of the recurrence.

    kwargs : dictionary
        Any extra parameters passed to :class:`lasagne.layers.Layer`
    """

    def __init__(self,
                 incoming,
                 num_x_to_h,
                 pre_compute_input=True,
                 combine_h_x=False,
                 no_bias=False,
                 learn_init=True,
                 post_indexes=None,
                 **kwargs):
        super(AbstractStepLayer, self).__init__(incoming, **kwargs)
        self.num_x_to_h = num_x_to_h
        self.pre_compute_input = pre_compute_input
        self.combine_h_x = combine_h_x
        self.no_bias = no_bias
        self.learn_init = learn_init
        self.post_indexes = post_indexes
        self.init = []

    def add_init_param(self, param, shape, name=None, f=None, **kwargs):
        if isinstance(param, T.TensorVariable):
            if param.ndim != len(shape):
                if param.ndim != len([s for s in shape if s is not None]):
                    raise ValueError(
                        "When {} is provided as a TensorVariable, it should"
                        " have shape {}".format(name, shape))
        else:
            param = self.add_param(param, utils.extract_clean_dims(shape),
                                   name=name, **kwargs)
        if f is not None:
            param = f(param)
        self.init.append((param, shape))
        return param

    def get_inits(self, ns):
        n = ns[0]
        inits = list()
        params = self.get_params()
        for param, shape in self.init:
            if param in params:
                pattern = ('x' if s is not None else 0 for s in shape)
                ones = T.ones((n,)).dimshuffle(*pattern)
                pattern = utils.broadcast_to_none(shape)
                p = ones * param.dimshuffle(*pattern)
                if not self.learn_init:
                    p = theano.gradient.zero_grad(p)
            elif param.ndim != len(shape):
                pattern = ('x' if s is not None else 0 for s in shape)
                ones = T.ones((n,)).dimshuffle(*pattern)
                pattern = utils.broadcast_to_none(shape)
                p = ones * param.dimshuffle(*pattern)
            else:
                p = param
            inits.append(p)
        return inits

    def get_output_shapes_for(self, input_shapes):
        raise NotImplementedError

    def get_outputs_for(self, inputs, **kwargs):
        raise NotImplementedError


class StandardStep(AbstractStepLayer):
    """
    lasagne.layers.recurrent.StandardStep(incoming, num_units,
    nonlinearity=nonlinearities.tanh, W=init.Orthogonal(),
    b=init.Constant(0.), h_init=init.Constant(0.), **kwargs)

    A step layer for a standard (vanilla) RNN.
    .. math ::
        h_t = \sigma(x_t W_x + h_{t-1} W_h + b)

    If precompute_input=False, then :math:`W` will be the concatenation
    of :math:`W_x` and :math:`W_h`.
    IF precompute_input=True, the bias :math:`b` is ignored as the one
    in the dense `in_to_hid` layer is used instead.


    Parameters
    ----------
    incoming : a tuple of either :class:`lasagne.layers.Layer`
        or tuples specifying the inputs shapes feeding into this layer.

    num_units : int
        Number of hidden units in the layer.

    nonlinearity : callable or None
        Nonlinearity to apply when computing new state (:math:`\sigma`). If
        None is provided, no nonlinearity will be applied.

    W : Theano shared variable, numpy array or callable
        Initializer for hidden-to-hidden weight matrix (:math:`W_h`).

    b : Theano shared variable, numpy array, callable or None
        Initializer for bias vector (:math:`b`). If None is provided there will
        be no bias.

    hid_init : callable, np.ndarray, theano.shared or :class:`Layer`
        Initializer for initial hidden state (:math:`h_0`).

    kwargs : dictionary
        Any extra parameters are passed to
        :class:`lasagne.layers.AbstractStepLayer`
    """
    def __init__(self, incoming, num_units,
                 nonlinearity=nonlinearities.tanh,
                 W=init.Orthogonal(),
                 b=init.Constant(0.),
                 h_init=init.Constant(0.),
                 **kwargs):
        super(StandardStep, self).__init__(incoming, num_units,
                                           combine_h_x=True, **kwargs)
        if len(self.input_shapes[0]) != 2:
            raise ValueError("StandardStep accepts only 2D inputs")
        if self.pre_compute_input:
            # If we precompute the output this is only dot(h, w)
            self.W = self.add_param(W, (num_units, num_units),
                                    name="W")
            self.b = None
        else:
            # If not we will concatenate x and h
            d = self.input_shapes[0][1]
            self.W = self.add_param(W, (d + num_units, num_units),
                                    name="W")
            self.b = None if b is None else self.add_param(b, (num_units, ),
                                                           name="b")
        self.add_init_param(h_init, (None, num_units), name="h_init")
        self.f = nonlinearity or nonlinearities.identity

    def get_output_shapes_for(self, input_shapes):
        x_shape = input_shapes[0]
        return (x_shape[0], self.num_x_to_h),

    def get_outputs_for(self, inputs, **kwargs):
        x = inputs[0]
        h = inputs[1]
        if self.pre_compute_input:
            # If we precompute the output this is only dot(h, w)
            return self.f(T.dot(h, self.W) + x),
        else:
            # If not we will concatenate x and h
            h = T.concatenate((x, h), axis=1)
            b = T.constant(0) if self.b is None else self.b.dimshuffle('x', 0)
            return self.f(T.dot(h, self.W) + b),


class GRUStep(AbstractStepLayer):
    """
    lasagne.layers.recurrent.GRUStep(incoming, num_units,
    nonlinearity=nonlinearities.tanh,
    gates_function=nonlinearities.sigmoid,
    W=init.Orthogonal(), h_init=init.Constant(0.), **kwargs)

    A GRU step layer
    .. math ::
        r_t &= \sigma_r(x_t W_{xr} + h_{t - 1} W_{hr} + b_r)\\
        u_t &= \sigma_u(x_t W_{xu} + h_{t - 1} W_{hu} + b_u)\\
        c_t &= \sigma_c(x_t W_{xc} + r_t \odot (h_{t - 1} W_{hc}) + b_c)\\
        h_t &= (1 - u_t) \odot h_{t - 1} + u_t \odot c_t

    It has no bias as it is always part of the `in_to_hid` layer.


    Parameters
    ----------
    incoming : a tuple of either :class:`lasagne.layers.Layer`
        or tuples specifying the inputs shapes feeding into this layer.

    num_units : int
        Number of hidden units in the layer.

    nonlinearity : callable or None
        Nonlinearity to apply when computing new state (:math:`\sigma_c`).
        If None is provided, no nonlinearity will be applied.

    gates_nonlinearity : callable or None
        Nonlinearity to apply for the gating mechanisms (:math:`\sigma_r`,
         :math:`\sigma_u`). If None is provided, no nonlinearity will
         be applied.

    W : Theano shared variable, numpy array or callable
        Initializer for hidden-to-hidden weight matrix (:math:`W_h`).

    hid_init : callable, np.ndarray, theano.shared or :class:`Layer`
        Initializer for initial hidden state (:math:`h_0`).

    kwargs : dictionary
        Any extra parameters are passed to
        :class:`lasagne.layers.AbstractStepLayer`
    """
    def __init__(self, incoming, num_units,
                 nonlinearity=nonlinearities.tanh,
                 gates_function=nonlinearities.sigmoid,
                 W=init.Orthogonal(),
                 h_init=init.Constant(0.),
                 **kwargs):
        super(GRUStep, self).__init__(incoming, 3 * num_units, **kwargs)
        if len(self.input_shapes[0]) != 2:
            raise ValueError("GRUStep accepts only 2D inputs")
        self.W = self.add_param(W, (num_units, 3 * num_units),
                                name="W")
        self.add_init_param(h_init, (None, num_units), name="h_init")
        self.f = nonlinearity or nonlinearities.identity
        self.g = gates_function or nonlinearities.identity

    def get_output_shapes_for(self, input_shapes):
        x_shape = input_shapes[0]
        return (x_shape[0], self.num_x_to_h // 3),

    def get_outputs_for(self, inputs, **kwargs):
        x = inputs[0]
        h = inputs[1]
        n = self.num_x_to_h // 3
        # If we precompute the output this is only dot(h, w)
        a = T.dot(h, self.W)
        ru = self.g(a[:, :2*n] + x[:, :2*n])
        r = ru[:, :n]
        u = ru[:, n:]
        c_in = a[:, 2*n:]
        c = self.f(c_in * r + x[:, 2*n:])
        return u * c + (1 - u) * h,


class LSTMStep(AbstractStepLayer):
    """
    lasagne.layers.recurrent.LSTMStep(incoming, num_units,
    nonlinearity=nonlinearities.tanh,
    gates_function=nonlinearities.sigmoid,
    W=init.Orthogonal(), b=init.Constant(),
    h_init=init.Constant(), c_init=init.Constant(),
    w_peep=init.GlorotUniform(), peepholes=False,
    post_indexes=(0, ), **kwargs)

    A LSTM step layer
    .. math ::

        i_t &= \sigma_i(x_t W_{xi} + h_{t-1} W_{hi}
               + w_{ci} \odot c_{t-1} + b_i)\\
        f_t &= \sigma_f(x_t W_{xf} + h_{t-1} W_{hf}
               + w_{cf} \odot c_{t-1} + b_f)\\
        c_t &= f_t \odot c_{t - 1}
               + i_t \odot \sigma_c(x_t W_{xc} + h_{t-1} W_{hc} + b_c)\\
        o_t &= \sigma_o(x_t W_{xo} + h_{t-1} W_{ho} + w_{co} \odot c_t + b_o)\\
        h_t &= o_t \odot \sigma_h(c_t)

    Parameters
    ----------
    incoming : a tuple of either :class:`lasagne.layers.Layer`
        or tuples specifying the inputs shapes feeding into this layer.

    num_units : int
        Number of hidden units in the layer.

    nonlinearity : callable or None
        Nonlinearity to apply when computing new state (:math:`\sigma_c`).
        If None is provided, no nonlinearity will be applied.

    gates_nonlinearity : callable or None
        Nonlinearity to apply for the gating mechanisms (:math:`\sigma_r`,
         :math:`\sigma_u`). If None is provided, no nonlinearity will
         be applied.

    W : Theano shared variable, numpy array or callable
        Initializer for hidden-to-hidden weight matrix (:math:`W_h`).

    b : Theano shared variable, numpy array, callable or None
        Initializer for bias vector (:math:`b`). If None is provided there will
        be no bias.

    h_init : callable, np.ndarray, theano.shared or :class:`Layer`
        Initializer for initial hidden state (:math:`h_0`).

    c_init : callable, np.ndarray, theano.shared or :class:`Layer`
        Initializer for initial hidden state (:math:`c_0`).

    w_peep: Theano shared variable, numpy array or callable
        Initializer for the peephole weights (:math:`w`).

    peephole: bool
        Whether to use peephole connections

    post_indexes: None or tuple of ints
        Filtering of the outputs

    kwargs : dictionary
        Any extra parameters are passed to
        :class:`lasagne.layers.AbstractStepLayer`
    """
    def __init__(self, incoming, num_units,
                 nonlinearity=nonlinearities.tanh,
                 gates_function=nonlinearities.sigmoid,
                 W=init.Orthogonal(),
                 b=init.Constant(),
                 h_init=init.Constant(),
                 c_init=init.Constant(),
                 w_peep=init.GlorotUniform(),
                 peepholes=False,
                 post_indexes=(0, ),
                 **kwargs):
        super(LSTMStep, self).__init__(incoming, 4 * num_units,
                                       combine_h_x=True,
                                       post_indexes=post_indexes, **kwargs)
        if len(self.input_shapes[0]) != 2:
            raise ValueError("LSTMStep accepts only 2D inputs")
        if self.pre_compute_input:
            # If we precompute the output this is only dot(h, w)
            self.W = self.add_param(W, (num_units, 4 * num_units),
                                    name="W")
            self.b = None
        else:
            # If not we will concatenate x and h
            d = self.input_shapes[0][1]
            self.W = self.add_param(W, (d + num_units, 4 * num_units),
                                    name="W")
            self.b = None if b is None else self.add_param(b, (4 * num_units,),
                                                           name="b")

        self.add_init_param(h_init, (None, num_units), name="h_init")
        self.add_init_param(c_init, (None, num_units), name="c_init")
        if peepholes:
            self.W_peep = self.add_param(w_peep, (num_units, 3 * num_units),
                                         name="W_peep")
        else:
            self.W_peep = None
        self.f = nonlinearity or nonlinearities.identity
        self.g = gates_function or nonlinearities.identity

    def get_output_shapes_for(self, input_shapes):
        x_shape = input_shapes[0]
        return (x_shape[0], self.num_x_to_h // 4), \
               (x_shape[0], self.num_x_to_h // 4)

    def get_outputs_for(self, inputs, **kwargs):
        x = inputs[0]
        h = inputs[1]
        c = inputs[2]
        n = self.num_x_to_h // 4
        if self.pre_compute_input:
            # If we precompute the output this is only dot(h, w)
            a = T.dot(h, self.W) + x
        else:
            # If not we will concatenate x and h
            h = T.concatenate((x, h), axis=1)
            b = T.constant(0) if self.b is None else self.b.dimshuffle('x', 0)
            a = T.dot(h, self.W) + b
        if self.W_peep is not None:
            peep_holes = c * self.W_peep.dimshuffle("x", 0)
            in_forget = self.g(a[:, :2*n] + peep_holes[:, :2*n])
            i = in_forget[:, :n]
            f = in_forget[:, n:]
            c = f * c + i * self.f(a[:, 2*n:3*n])
            o = self.g(a[:, 3*n:] + peep_holes[:, 2*n:])
            h = o * self.f(c)
            return h, c
        else:
            in_forget = self.g(a[:, :2 * n])
            i = in_forget[:, :n]
            f = in_forget[:, n:]
            c = f * c + i * self.f(a[:, 2 * n:3 * n])
            o = self.g(a[:, 3 * n:])
            h = o * self.f(c)
            return h, c


class RWAStep(AbstractStepLayer):
    """
    lasagne.layers.recurrent.LSTMStep(incoming, num_units,
    bounding_nonlinearity=nonlinearities.tanh,
    nonlinearity=nonlinearities.tanh,
    W=init.Orthogonal(), h_init=init.Normal(std=1.0),
    post_indexes=(0,), **kwargs)

    A Recurrence Weighted Average step layer
    .. math ::

    Source: https://arxiv.org/abs/1703.01253

    Parameters
    ----------
    incoming : a tuple of either :class:`lasagne.layers.Layer`
        or tuples specifying the inputs shapes feeding into this layer.

    num_units : int
        Number of hidden units in the layer.

    bounding_nonlinearity : callable or None
        Nonlinearity to apply for bounding :math:`h_t`.
        If None is provided, no nonlinearity will be applied.

    nonlinearity : callable or None
        Nonlinearity to apply when computing new state (:math:`\sigma_c`).
        If None is provided, no nonlinearity will be applied.

    gates_nonlinearity : callable or None
        Nonlinearity to apply for the gating mechanisms (:math:`\sigma_r`,
         :math:`\sigma_u`). If None is provided, no nonlinearity will
         be applied.

    W : Theano shared variable, numpy array or callable
        Initializer for hidden-to-hidden weight matrix (:math:`W_h`).

    h_init : callable, np.ndarray, theano.shared or :class:`Layer`
        Initializer for initial hidden state (:math:`h_0`).

    post_indexes: None or tuple of ints
        Filtering of the outputs

    kwargs : dictionary
        Any extra parameters are passed to
        :class:`lasagne.layers.AbstractStepLayer`
    """
    def __init__(self, incoming, num_units,
                 bounding_nonlinearity=nonlinearities.tanh,
                 nonlinearity=nonlinearities.tanh,
                 W=init.Orthogonal(),
                 h_init=init.Normal(std=1.0),
                 post_indexes=(0,),
                 **kwargs):
        super(RWAStep, self).__init__(incoming, 3 * num_units,
                                      post_indexes=post_indexes,
                                      **kwargs)
        if len(self.input_shapes[0]) != 2:
            raise ValueError("GRUStep accepts only 2D inputs")
        # dot(h, W)
        self.W = self.add_param(W, (num_units, 2 * num_units), name="W")
        # functions
        self.f = nonlinearity
        self.g = bounding_nonlinearity
        # Initial condition
        self.h_init = self.add_init_param(h_init, (None, num_units),
                                          name="h_init", f=self.g)
        zeros = T.zeros((num_units, ))
        # nominator
        self.add_init_param(zeros, (None, num_units))
        # denominator
        self.add_init_param(zeros, (None, num_units))

    def get_output_shapes_for(self, input_shapes):
        x_shape = input_shapes[0]
        return (x_shape[0], self.num_x_to_h // 3), \
               (x_shape[0], self.num_x_to_h // 3), \
               (x_shape[0], self.num_x_to_h // 3)

    def get_outputs_for(self, inputs, **kwargs):
        x = inputs[0]
        h = inputs[1]
        nt = inputs[2]
        dt = inputs[3]
        # dimensionality
        n = self.num_x_to_h // 3
        # eq. 4 from the paper
        u = x[:, :n]
        ga = T.dot(h, self.W) + x[:, n:]
        g = ga[:, :n]
        a = ga[:, n:]
        # eq. 3 from the paper
        z = u * self.g(g)
        # conditioning the normalization
        m = T.maximum(a, T.log(dt))
        nt = nt * T.exp(- m) + z * T.exp(a - m)
        dt = dt * T.exp(- m) + T.exp(a - m)
        h = self.f(nt / dt)
        return h, nt, dt

