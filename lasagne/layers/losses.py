import theano.tensor as T

from .. import utils
from ..layers import Layer
from .. import random as rnd
from ..nonlinearities import softplus, identity

__all__ = [
    "SKFGNLossLayer",
    "SumLosses",
    "SquaredError",
    "BinaryLogitsCrossEntropy",
    "bernoulli_logits_ll",
    "CategoricalLogitsCrossEntropy",
    "categorical_logits_ll",
    "BetaCrossEntropy",
    "beta_ll"
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

    def curvature_propagation(self, optimizer, inputs, outputs, curvature, make_matrix):
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
            return sum(utils.expand_variable(i, r) * w for i, w, r in
                       zip(inputs, self.weights, self.repeats)),
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

    def curvature_propagation(self, optimizer, inputs, outputs, curvature, make_matrix):
        return [T.abs_(T.constant(w)) for w in self.weights]

    def gauss_newton_product(self, inputs_map, outputs_map, params, variant, v1, v2=None):
        for l in self.input_layers:
            assert isinstance(l, SKFGNLossLayer)
        return sum(l_in.gauss_newton_product(inputs_map, outputs_map,
                                             params, variant, v1, v2) * w
                   for l_in, w in zip(self.input_layers, self.skfgn_weights))


class SquaredError(SKFGNLossLayer):
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
        super(SquaredError, self).__init__((x, y),
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

    def curvature_propagation(self, optimizer, inputs, outputs, curvature, make_matrix):
        x, y = inputs
        assert len(curvature) == 1
        weight = T.sqrt(curvature[0])
        if optimizer.variant == "skfgn-rp":
            cl_x = optimizer.random_sampler(x.shape)
            cl_x *= weight
            cl_y = optimizer.random_sampler(y.shape)
            cl_y *= weight
            return cl_x, cl_y
        elif optimizer.variant == "skfgn-fisher" or optimizer.variant == "kfac*":
            fake_dx = rnd.normal(x.shape)
            fake_dx *= weight
            fake_dy = rnd.normal(y.shape)
            fake_dy *= weight
            return fake_dx, fake_dy
        elif optimizer.variant == "kfra":
            if x.ndim == 1:
                gn_x = weight
                gn_y = weight
            elif x.ndim == 2:
                x = T.flatten(x, outdim=2)
                y = T.flatten(y, outdim=2)
                gn_x = T.eye(x.shape[1])
                gn_x *= curvature[0]
                gn_y = T.eye(y.shape[1])
                gn_y *= curvature[0]
            else:
                raise ValueError("KFRA can not work on more than 2 dimensions.")
            return gn_x, gn_y
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

    def curvature_propagation(self, optimizer, inputs, outputs, curvature, make_matrix):
        x, y = inputs
        assert len(curvature) == 1
        weight = T.sqrt(curvature[0])
        p_x = T.nnet.sigmoid(x)
        if optimizer.variant == "skfgn-rp":
            v = optimizer.random_sampler(x.shape)
            cl_v = T.sqrt(p_x * (1 - p_x)) * v
            cl_v *= weight
            return cl_v, T.zeros_like(y)
        elif optimizer.variant == "skfgn-fisher" or optimizer.variant == "kfac*":
            fake_dx = p_x - utils.th_fx(rnd.binary(x.shape, p=p_x))
            fake_dx *= weight
            return fake_dx, T.zeros_like(y)
        elif optimizer.variant == "kfra":
            if x.ndim == 1:
                gn_x = T.diag(p_x * (1 - p_x))
            else:
                p_x = T.flatten(p_x, outdim=2)
                gn_x = T.diag(T.mean(p_x * (1 - p_x), axis=0))
            gn_x *= weight**2
            return gn_x, T.constant(0)
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


class CategoricalLogitsCrossEntropy(SKFGNLossLayer):
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
        super(CategoricalLogitsCrossEntropy, self).__init__((output, target),
                                                            (x_repeats, y_repeats),
                                                            max_inputs=2, *args, **kwargs)

    def get_outputs_for(self, inputs, **kwargs):
        assert len(inputs) == 2
        x, y = inputs
        if x.ndim != 2:
            x = T.flatten(x, outdim=2)
        x = utils.expand_variable(x, self.repeats[0])
        y = utils.expand_variable(y, self.repeats[1])
        p_x = T.nnet.softmax(x)
        # Calculate loss
        cce = T.nnet.categorical_crossentropy(p_x, y)
        # Calculate per data point
        return cce,

    def curvature_propagation(self, optimizer, inputs, outputs, curvature, make_matrix):
        x, y = inputs
        assert len(curvature) == 1
        weight = curvature[0]
        if x.ndim != 2:
            p_x = T.nnet.softmax(T.flatten(x, outdim=2))
        else:
            p_x = T.nnet.softmax(x)
        if optimizer.variant == "skfgn-rp":
            v = optimizer.random_sampler(x.shape)
            cl_v = T.sqrt(p_x * (1 - p_x)) * v
            cl_v *= T.sqrt(weight)
            if x.ndim != 2:
                cl_v = T.reshape(cl_v, x.shape)
            return cl_v, T.zeros_like(y)
        elif optimizer.variant == "skfgn-fisher" or optimizer.variant == "kfac*":
            fake_dx = p_x - utils.th_fx(rnd.multinomial(x.shape, pvals=p_x))
            fake_dx *= T.sqrt(weight)
            if x.ndim != 2:
                fake_dx = T.reshape(fake_dx, x.shape)
            return fake_dx, T.zeros_like(y)
        elif optimizer.variant == "kfra":
            gn_x = T.diag(T.mean(p_x, axis=0)) - T.dot(p_x.T, p_x) / utils.th_fx(p_x.shape[0])
            gn_x *= weight
            return gn_x, T.constant(0)
        else:
            raise ValueError("Unreachable!")

    def gauss_newton_product(self, inputs_map, outputs_map, params, variant, v1, v2=None):
        x, y = inputs_map[self]
        h = T.nnet.softmax(x)

        Jv1 = T.Rop(x, params, v1)
        Jv2 = T.Rop(x, params, v2) if v2 else Jv1

        # The Hessian is diag(h) - h h^T
        # (diag(h) - h h^T) v = h * v - h h^T v = h * (v - h^T v) = h * v2
        # v1^T H v2 = sum(v1 * h * v2)
        v2 = Jv2 - T.sum(Jv2 * h, axis=1).dimshuffle(0, 'x')
        return T.mean(T.sum(Jv1 * h * v2, axis=1))


def categorical_logits_ll(output, target,
                          x_repeats=1, y_repeats=1,
                          *args, **kwargs):
    layer = CategoricalLogitsCrossEntropy(output, target,
                                          x_repeats, y_repeats,
                                          *args, **kwargs)
    return SumLosses(layer, mode="merge", weights=[-1])


class BetaCrossEntropy(SKFGNLossLayer):
    """
    Calculates the beta cross entropy between a Beta distribution and
    observations (these should be in the interval [0, 1]).

    The `params` of the Beta distribution are assumed to be the
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
    def __init__(self, params, observations,
                 min_params=1.0,
                 nonlinearity=softplus,
                 epsilon=1e-6,
                 x_repeats=1, y_repeats=1, *args, **kwargs):
        super(BetaCrossEntropy, self).__init__((params, observations),
                                               (x_repeats, y_repeats),
                                               max_inputs=2, *args, **kwargs)
        self.min_params = T.constant(min_params)
        self.nonlinearity = nonlinearity
        self.epsilon = epsilon

    def get_outputs_for(self, inputs, **kwargs):
        assert len(inputs) == 2
        x, y = inputs
        assert x.ndim == y.ndim
        x = self.nonlinearity(x) + self.min_params + self.epsilon
        x = utils.expand_variable(x, self.repeats[0])
        y = utils.expand_variable(y, self.repeats[1])
        d = T.int_div(x.shape[1], 2)
        alpha = x[:, :d]
        beta = x[:, d:]
        # Calculate loss in a stable way
        c = alpha + beta - 2
        r = (alpha - 1) / c
        y = T.clip(y, self.epsilon, 1.0 - self.epsilon)
        loss = c * T.nnet.binary_crossentropy(y, r)
        # Add the beta function values
        loss += T.gammaln(alpha) + T.gammaln(beta) - T.gammaln(alpha + beta)
        # Flatten
        loss = T.flatten(loss, outdim=2)
        # Calculate per data point
        loss = T.sum(loss, axis=1)
        return loss,

    def curvature_propagation(self, optimizer, inputs, outputs, curvature, make_matrix):
        x, y = inputs
        assert len(curvature) == 1
        weight = T.sqrt(curvature[0])
        x_f = self.nonlinearity(x) + self.min_params
        d = T.int_div(x.shape[1], 2)
        alpha = x_f[:, :d]
        beta = x_f[:, d:]
        if optimizer.variant == "skfgn-rp":
            va, vb = optimizer.random_sampler((alpha.shape, beta.shape))
            da = T.tri_gamma(alpha)
            db = T.tri_gamma(beta)
            dab = T.tri_gamma(alpha + beta)
            chol_x = T.sqrt(da - dab)
            chol_y = - dab / chol_x
            chol_z = T.sqrt(db - dab - T.sqr(chol_y))
            v1 = chol_x * va
            v2 = chol_y * va + chol_z * vb
            cl_v = T.concatenate((v1, v2), axis=1)
            cl_v *= weight
            return T.Lop(x_f, x, cl_v), T.zeros_like(y)
        elif optimizer.variant == "skfgn-fisher" or optimizer.variant == "kfac*":
            alpha = utils.expand_variable(alpha, self.repeats[0])
            beta = utils.expand_variable(beta, self.repeats[0])
            fake_y = rnd.beta(alpha, beta)
            fake_y = T.clip(fake_y, self.epsilon, 1.0 - self.epsilon)
            fake_da = T.log(fake_y) - T.psi(alpha) + T.psi(alpha + beta)
            fake_db = T.log(1.0 - fake_y) - T.psi(beta) + T.psi(alpha + beta)
            fake_dx = T.concatenate((fake_da, fake_db), axis=1)
            fake_dx *= weight
            return fake_dx, T.zeros_like(y)
        elif optimizer.variant == "kfra":
            da = T.mean(T.tri_gamma(alpha), axis=0)
            db = T.mean(T.tri_gamma(beta), axis=0)
            dab = T.mean(T.tri_gamma(alpha + beta), axis=0)
            gn_x1 = T.concatenate((T.diag(da), T.diag(dab)), axis=0)
            gn_x2 = T.concatenate((T.diag(dab), T.diag(db)), axis=0)
            gn_x = T.concatenate((gn_x1, gn_x2), axis=1)
            if self.nonlinearity != identity:
                a = T.Lop(outputs[0], x_f, T.ones_like(x))
                gn_x *= T.dot(a.T, a) / utils.th_fx(x.shape[0])
            return gn_x, T.constant(0)
        else:
            raise ValueError("Unreachable!")

    def gauss_newton_product(self, inputs_map, outputs_map, params, variant, v1, v2=None):
        x, y = inputs_map[self]
        x_f = self.nonlinearity(x) + self.min_params
        Jv1 = utils.Rop(x_f, params, v1)
        Jv2 = utils.Rop(x_f, params, v2) if v2 else Jv1
        d = T.int_div(x.shape[1], 2)
        alpha = x_f[:, :d]
        beta = x_f[:, d:]
        v1a = Jv1[:, :d]
        v1b = Jv1[:, d:]
        v2a = Jv2[:, :d]
        v2b = Jv2[:, d:]
        da = T.mean(T.tri_gamma(alpha), axis=0)
        db = T.mean(T.tri_gamma(beta), axis=0)
        dab = T.mean(T.tri_gamma(alpha + beta), axis=0)
        gn = v1a * (da - dab) * v2a - dab * (v1a * v2b + v1b * v2a) + v1b * (db - dab) * v2b
        return T.mean(T.sum(gn, axis=1), axis=0)


def beta_ll(output, target, x_repeats=1, y_repeats=1, *args, **kwargs):
    layer = BetaCrossEntropy(output, target,
                             x_repeats=x_repeats,
                             y_repeats=y_repeats,
                             *args, **kwargs)
    return SumLosses(layer, mode="merge", weights=[-1])
