import theano.tensor as T

from .. import utils
from ..layers import Layer
from ..random import normal, binary

__all__ = [
    "SKFGNLossLayer",
    "SumLosses",
    "SquaredLoss",
    "BinaryLogitsCrossEntropy",
    "bernoulli_logits_ll"
]


class SKFGNLossLayer(Layer):
    """
    The :class:`SKFGNLossLayer` class represents a loss, which can be used
    with Kronecker-Factored methods (SKFGN and KFRA).
    It should be subclassed when implementing new types of losses.

    All losses assume that the instance dimension is the first and sum over
    other dimensions. This means that the shape of the output is always (N, ).

    Parameters
    ----------
    incoming : a tuple or list of :class:`Layer` or a single instance or shape.
        The layer/s feeding into this layer.

    repeats : tuple or list of ints
        If when calculating the loss any of the layers need to be repeat,
        this will contain how many time to repeat each incoming layer.
        The length of `repeats` must be equal to the number of inputs.

    Any other arguments are passed to the base :class: `Layer`.
    """
    def __init__(self, incoming, repeats=(1, ), *args, **kwargs):
        super(SKFGNLossLayer, self).__init__(incoming, *args, **kwargs)
        if len(repeats) == 1:
            self.repeats = tuple(repeats[0] for _ in self.input_shapes)
        else:
            assert len(repeats) == len(self.input_shapes)
            self.repeats = repeats

    def get_output_shapes_for(self, input_shapes):
        assert len(input_shapes) == len(self.input_shapes)
        known_shapes = []
        for in_shape, r in zip(input_shapes, self.repeats):
            if in_shape[0] is not None:
                known_shapes.append(in_shape[0] * r)
        if len(known_shapes) == 0:
            return (None, ),
        else:
            for s in known_shapes[1:]:
                assert s == known_shapes[0]
            return (known_shapes[0], ),

    def get_outputs_for(self, inputs, **kwargs):
        raise NotImplementedError

    def skfgn(self, optimizer, inputs, outputs, curvature, kronecker_inversion):
        raise NotImplementedError

    def gauss_newton_product(self, inputs_map, outputs_map, params, variant, v1, v2=None):
        raise NotImplementedError


class SumLosses(SKFGNLossLayer):
    """
    A layer for combining multiple loss layers together by summation over them.

    Parameters
    ----------
    incoming : tuple or list or ndarray or None
        How to weight each incoming layer. If none all weights are set to 1.

    mode : str (default: mean)
        1. "mean" - sums the means of all incoming losses
        2. "sum" - sums the sums of all incoming losses
        3. "merge" - just adds together all incoming losses without reductions

    weights : tuple or list of ints
        If when calculating the loss any of the layers need to be repeat,
        this will contain how many time to repeat each incoming layer.
        The length of `repeats` must be equal to the number of inputs.

    skfgn_weights : tuple or list of ints
        If for some reason you want to weight differently the losses for
        the curvature calculation you can use this. By default these are
        initialized to `weights`
    """

    def __init__(self, incoming, mode="merge",
                 weights=None, skfgn_weights=None, **kwargs):
        assert mode in ("merge", "sum", "mean")
        self.mode = mode
        super(SumLosses, self).__init__(incoming, max_inputs=10, **kwargs)
        if weights is None:
            self.weights = [T.constant(1) for _ in self.input_shapes]
        else:
            self.weights = weights
        if skfgn_weights is None:
            self.skfgn_weights = self.weights
        else:
            self.skfgn_weights = skfgn_weights

    @staticmethod
    def get_shape_on_axis(shapes, axis=0):
        n = None
        for shape in shapes:
            if len(shape) > axis and shape[axis] is not None:
                if n is not None:
                    assert n == shape[axis]
                else:
                    n = shape[axis]
        return n

    def get_outputs_for(self, inputs, **kwargs):
        assert len(inputs) == len(self.weights)
        if self.mode == "merge":
            return sum(i * w for i, w in zip(inputs, self.weights)),
        elif self.mode == "mean":
            return sum(T.mean(i) * w for i, w in zip(inputs, self.weights)),
        else:
            return sum(T.sum(i) * w for i, w in zip(inputs, self.weights)),

    def get_output_shapes_for(self, input_shapes):
        if self.mode == "merge":
            n = SumLosses.get_shape_on_axis(input_shapes)
            return (n, ),
        else:
            return (),

    def skfgn(self, optimizer, inputs, outputs, curvature, kronecker_inversion):
        return self.weights

    def gauss_newton_product(self, inputs_map, outputs_map, params, variant, v1, v2=None):
        for l in self.input_layers:
            assert isinstance(l, SKFGNLossLayer)
        return sum(l_in.gauss_newton_product(inputs_map, outputs_map,
                                             params, variant, v1, v2) * w
                   for l_in, w in zip(self.input_layers, self.skfgn_weights))


class SquaredLoss(SKFGNLossLayer):
    """
    A squared loss layer calculating:
        0.5 (x - y)^T (x - y)

    If x and y are of more than 2D dimensions they are first flattened to 2D.

    Parameters
    ----------
    x : a :class:`Layer` instance or a tuple or list
        The layer or the expected input shape for `x`.

    y : a :class:`Layer` instance or a tuple or list
        The layer or the expected input shape for `y`.

    x_repeats: int (default: 1)
        If `x` has to be repeated some number of times.

    y_repeats: int (default: 1)
        If `y` has to be repeated some number of times.
    """

    def __init__(self, x, y, x_repeats=1, y_repeats=1, *args, **kwargs):
        super(SquaredLoss, self).__init__((x, y),
                                          (x_repeats, y_repeats),
                                          max_inputs=2, *args, **kwargs)
        assert len(self.input_shapes) == 2

    def get_outputs_for(self, inputs, **kwargs):
        assert len(inputs) == 2
        x, y = inputs
        assert x.ndim == y.ndim
        x = T.flatten(x, outdim=2) if x.ndim > 2 else x
        y = T.flatten(y, outdim=2) if y.ndim > 2 else y
        x = utils.expand_variable(x, self.repeats[0])
        y = utils.expand_variable(y, self.repeats[1])
        if x.ndim == 1:
            return T.sqr(x - y) / 2,
        else:
            return T.sum(T.sqr(x - y), axis=1) / 2,

    def skfgn(self, optimizer, inputs, outputs, curvature, kronecker_inversion):
        x, y = inputs
        assert len(curvature) == 1
        weight = curvature[0]
        if optimizer.variant == "skfgn-rp":
            cl_v = optimizer.random_sampler(x.shape)
            cl_v *= T.sqrt(weight)
            return cl_v, None
        elif optimizer.variant == "skfgn-fisher" or optimizer.variant == "kfac*":
            fake_dx = normal(x.shape)
            fake_dx *= T.sqrt(weight)
            return fake_dx, None
        elif optimizer.variant == "kfra":
            if x.ndim == 1:
                gn_x = T.sqrt(weight)
            elif x.ndim == 2:
                x = T.flatten(x, outdim=2)
                gn_x = T.eye(x.shape[1])
                gn_x *= T.sqrt(weight)
            else:
                raise ValueError("KFRA can not work on more than 2 dimensions.")
            return gn_x, None
        else:
            raise ValueError("Unreachable!")

    def gauss_newton_product(self, inputs_map, outputs_map, params, variant, v1, v2=None):
        x, y = inputs_map[self]
        return utils.gauss_newton_product(x, T.constant(1), params, v1, v2)


class BinaryLogitsCrossEntropy(SKFGNLossLayer):
    """
    Calculates the binary cross entropy between `output`(x) and `target`(y).
    The `output`(x) are assumed to be the logits of the Bernoulli distribution.
    Formally computes:
        - y log(sigmoid(x)) - (1 - y) * log(sigmoid(-x))

    Note that curvature is propagate only to `output`(x).

    Parameters
    ----------
    output : a :class:`Layer` instance or a tuple or list
        The layer or the expected input shape for `output`.

    target : a :class:`Layer` instance or a tuple or list
        The layer or the expected input shape for `target`.

    x_repeats: int (default: 1)
        If `output` has to be repeated some number of times.

    y_repeats: int (default: 1)
        If `target` has to be repeated some number of times.
    """
    def __init__(self, output, target, x_repeats=1, y_repeats=1, *args, **kwargs):
        super(BinaryLogitsCrossEntropy, self).__init__((output, target),
                                                       (x_repeats, y_repeats),
                                                       max_inputs=2, *args, **kwargs)

    def get_outputs_for(self, inputs, **kwargs):
        assert len(inputs) == 2
        x, y = inputs
        assert x.ndim == y.ndim
        x = utils.expand_variable(x, self.repeats[0])
        y = utils.expand_variable(y, self.repeats[1])
        p_x = T.nnet.sigmoid(x)
        # Calculate loss
        bce = T.nnet.binary_crossentropy(p_x, y)
        # Flatten
        bce = T.flatten(bce, outdim=2)
        # Calculate per data point
        bce = T.sum(bce, axis=1)
        return bce,

    def skfgn(self, optimizer, inputs, outputs, curvature, kronecker_inversion):
        x, y = inputs
        assert len(curvature) == 1
        weight = curvature[0]
        p_x = T.nnet.sigmoid(x)
        if optimizer.variant == "skfgn-rp":
            v = optimizer.random_sampler(x.shape)
            cl_v = T.sqrt(p_x * (1 - p_x)) * v
            cl_v *= T.sqrt(weight)
            return cl_v, None
        elif optimizer.variant == "skfgn-fisher" or optimizer.variant == "kfac*":
            fake_dx = p_x - utils.th_fx(binary(x.shape, p=p_x))
            fake_dx *= T.sqrt(weight)
            return fake_dx, None
        elif optimizer.variant == "kfra":
            if x.ndim == 1:
                gn_x = T.diag(p_x * (1 - p_x))
            else:
                p_x = T.flatten(p_x, outdim=2)
                gn_x = T.diag(T.mean(p_x * (1 - p_x), axis=0))
            gn_x *= T.sqrt(weight)
            return gn_x, None
        else:
            raise ValueError("Unreachable!")

    def gauss_newton_product(self, inputs_map, outputs_map, params, variant, v1, v2=None):
        x, y = inputs_map[self]
        sigmoid = T.nnet.sigmoid(x)
        hess = sigmoid * (1 - sigmoid)
        return utils.gauss_newton_product(x, hess, params, v1, v2)


def bernoulli_logits_ll(output, target,
                        x_repeats=1, y_repeats=1,
                        *args, **kwargs):
    layer = BinaryLogitsCrossEntropy(output, target,
                                     x_repeats, y_repeats,
                                     *args, **kwargs)
    return SumLosses(layer, mode="merge", weights=[-1])
