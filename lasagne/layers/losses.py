import theano.tensor as T
from ..layers import Layer
from ..utils import gauss_newton_product, th_fx, repeat_variable
from ..random import th_normal, th_binary

__all__ = [
    "KFLossLayer",
    "SquareLoss",
    "BinaryLogitsCrossEntropy"
]


class KFLossLayer(Layer):
    def __init__(self, incoming, x_repeats=1, y_repeats=1, *args, **kwargs):
        super(KFLossLayer, self).__init__(incoming, *args, **kwargs)
        self.x_repeats = x_repeats
        self.y_repeats = y_repeats

    def get_output_shapes_for(self, input_shapes):
        assert len(input_shapes) == len(self.input_shapes)
        if input_shapes[0][0] is not None:
            return (input_shapes[0][0] * self.x_repeats, ),
        elif input_shapes[1][0] is not None:
            return (input_shapes[1][0] * self.y_repeats, ),
        else:
            return (None, ),

    def get_outputs_for(self, inputs, **kwargs):
        raise NotImplementedError

    def skfgn(self, optimizer, inputs, outputs, curvature, kronecker_inversion):
        raise NotImplementedError

    def gauss_newton_product(self, inputs, outputs, params, v1, v2=None):
        raise NotImplementedError


class SquareLoss(KFLossLayer):
    """
    Squared loss = 0.5 (x - y)^T (x - y)
    """

    def __init__(self, incoming, x_repeats=1, y_repeats=1, *args, **kwargs):
        super(SquareLoss, self).__init__(incoming, x_repeats, y_repeats,
                                         max_inputs=2, *args, **kwargs)
        assert len(self.input_shapes) == 2

    def get_outputs_for(self, inputs, **kwargs):
        assert len(inputs) == 2
        x, y = inputs
        assert x.ndim == y.ndim
        x = repeat_variable(x, self.x_repeats)
        y = repeat_variable(y, self.y_repeats)
        if x.ndim == 1:
            return T.sqr(x - y) / 2,
        elif x.ndim == 2:
            return T.sum(T.sqr(x - y), axis=1) / 2,
        elif x.ndim == 2:
            return T.sum(T.sqr(x - y), axis=(1, 2)) / 2,
        else:
            return T.sum(T.sqr(x - y), axis=(1, 2, 3)) / 2,

    def skfgn(self, optimizer, inputs, outputs, curvature, kronecker_inversion):
        x, y = inputs
        assert len(curvature) == 1
        weight = curvature[0]
        if optimizer.variant == "skfgn_rp":
            cl_v = optimizer.random_sampler(x.shape)
            cl_v *= T.sqrt(weight)
            return cl_v, None
        elif optimizer.variant == "skfgn_fisher" or optimizer.variant == "kfac*":
            fake_dx = th_normal(x.shape)
            fake_dx *= T.sqrt(weight)
            return fake_dx, None
        elif optimizer.variant == "skfgn_i":
            raise NotImplementedError
        elif optimizer.variant == "kfra":
            if x.ndim == 1:
                gn_x = T.sqrt(weight)
            else:
                x = T.flatten(x, outdim=2)
                gn_x = T.eye(x.shape[1])
                gn_x *= T.sqrt(weight)
            return gn_x, None
        else:
            raise ValueError("Unreachable")

    def gauss_newton_product(self, inputs, outputs, params, v1, v2=None):
        x, y = inputs
        return gauss_newton_product(x, 1, params, v1, v2)


class BinaryLogitsCrossEntropy(KFLossLayer):
    """
    Binary logits cross entropy = - y log(sigmoid(x)) - (1 - y) * log(sigmoid(-x))
    """
    def __init__(self, incoming, x_repeats=1, y_repeats=1, *args, **kwargs):
        super(BinaryLogitsCrossEntropy, self).__init__(incoming, x_repeats, y_repeats,
                                                       max_inputs=2, *args, **kwargs)

    def get_outputs_for(self, inputs, **kwargs):
        assert len(inputs) == 2
        x, y = inputs
        assert x.ndim == y.ndim
        x = repeat_variable(x, self.x_repeats)
        y = repeat_variable(y, self.y_repeats)
        p_x = T.nnet.sigmoid(x)
        bce = T.nnet.binary_crossentropy(p_x, y)
        bce = T.flatten(bce, outdim=2)
        return T.sum(bce, axis=1),

    def skfgn(self, optimizer, inputs, outputs, curvature, kronecker_inversion):
        x, y = inputs
        assert len(curvature) == 1
        weight = curvature[0]
        p_x = T.nnet.sigmoid(x)
        if optimizer.variant == "skfgn_rp":
            v = optimizer.random_sampler(x.shape)
            cl_v = T.sqrt(p_x * (1 - p_x)) * v
            cl_v *= T.sqrt(weight)
            return cl_v, None
        elif optimizer.variant == "skfgn_fisher" or optimizer.variant == "kfac*":
            fake_dx = p_x - th_fx(th_binary(x.shape, p=p_x))
            fake_dx *= T.sqrt(weight)
            return fake_dx, None
        elif optimizer.variant == "skfgn_i":
            raise NotImplementedError
        elif optimizer.variant == "kfra":
            if x.ndim == 1:
                gn_x = T.diag(p_x * (1 - p_x))
            else:
                p_x = T.flatten(p_x, outdim=2)
                gn_x = T.diag(T.mean(p_x * (1 - p_x), axis=0))
            gn_x *= T.sqrt(weight)
            return gn_x, None
        else:
            raise ValueError("Unreachable")

    def gauss_newton_product(self, inputs, outputs, params, v1, v2=None):
        x, y = inputs
        sigmoid = T.nnet.sigmoid(x)
        hess = sigmoid * (1 - sigmoid)
        return gauss_newton_product(x, hess, params, v1, v2)



