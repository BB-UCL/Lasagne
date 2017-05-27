import theano.tensor as T

from .. import utils
from ..layers import Layer
from ..random import normal, binary

__all__ = [
    "SKFGNLossLayer",
    "SumLossesLayer",
    "SquareLoss",
    "BinaryLogitsCrossEntropy",
    "GaussianKL"
]


class SKFGNLossLayer(Layer):
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


class SumLossesLayer(SKFGNLossLayer):
    """
        Binary cross entropy = - y log(sigmoid(x)) - (1 - y) * log(sigmoid(-x))
        """

    def __init__(self, incoming, weights=None, report_weights=None, **kwargs):
        super(SumLossesLayer, self).__init__(incoming, max_inputs=10, **kwargs)
        if weights is None:
            self.weights = [T.constant(1) for _ in self.input_shapes]
        else:
            self.weights = weights
        if report_weights is None:
            self.report_weights = self.weights
        else:
            self.report_weights = report_weights

    def get_outputs_for(self, inputs, **kwargs):
        assert len(inputs) == len(self.weights)
        return sum(T.mean(i) * w for i, w in zip(inputs, self.report_weights)),

    def get_output_shapes_for(self, input_shapes):
        return ()

    def skfgn(self, optimizer, inputs, outputs, curvature, kronecker_inversion):
        return self.weights

    def gauss_newton_product(self, inputs_map, outputs_map, params, variant, v1, v2=None):
        for l in self.input_layers:
            assert isinstance(l, SKFGNLossLayer)
        return sum(l_in.gauss_newton_product(inputs_map, outputs_map, params, variant, v1, v2) * w
                   for l_in, w in zip(self.input_layers, self.weights))


class SquareLoss(SKFGNLossLayer):
    """
    Squared loss = 0.5 (x - y)^T (x - y)
    """

    def __init__(self, incoming, x_repeats=1, y_repeats=1, *args, **kwargs):
        super(SquareLoss, self).__init__(incoming, (x_repeats, y_repeats),
                                         max_inputs=2, *args, **kwargs)
        assert len(self.input_shapes) == 2

    def get_outputs_for(self, inputs, **kwargs):
        assert len(inputs) == 2
        x, y = inputs
        assert x.ndim == y.ndim
        x = utils.expand_variable(x, self.repeats[0])
        y = utils.expand_variable(y, self.repeats[1])
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
        if optimizer.variant == "skfgn-rp":
            cl_v = optimizer.random_sampler(x.shape)
            cl_v *= T.sqrt(weight)
            return cl_v, None
        elif optimizer.variant == "skfgn-fisher" or optimizer.variant == "kfac*":
            fake_dx = normal(x.shape)
            fake_dx *= T.sqrt(weight)
            return fake_dx, None
        # elif optimizer.variant == "skfgn-i":
        #     raise NotImplementedError
        elif optimizer.variant == "kfra":
            if x.ndim == 1:
                gn_x = T.sqrt(weight)
            else:
                x = T.flatten(x, outdim=2)
                gn_x = T.eye(x.shape[1])
                gn_x *= T.sqrt(weight)
            return gn_x, None
        else:
            raise ValueError("Unreachable!")

    def gauss_newton_product(self, inputs_map, outputs_map, params, variant, v1, v2=None):
        x, y = inputs_map[self]
        return utils.gauss_newton_product(x, T.constant(1), params, v1, v2)


class BinaryLogitsCrossEntropy(SKFGNLossLayer):
    """
    Binary logits cross entropy = - y log(sigmoid(x)) - (1 - y) * log(sigmoid(-x))
    """
    def __init__(self, incoming, x_repeats=1, y_repeats=1, *args, **kwargs):
        super(BinaryLogitsCrossEntropy, self).__init__(incoming, (x_repeats, y_repeats),
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
        # Calculat per data point
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
        # elif optimizer.variant == "skfgn-i":
        #     raise NotImplementedError
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


class GaussianKL(SKFGNLossLayer):
    """
    Calculates the KL(q||p) where q and p are isotropic Gaussian distributions.

    !!! The arguments parametrize the mean and the standard deviation not the variance!!!

    $$ KL(q||p) = \frac{1}{2} \sum_i \left( \frac{\sigma_{q_i}^2}{\sigma_{p_i}^2} + 
    \frac{(\mu_{p_i} - \mu_{q_i})^2}{\sigma_{p_i}^2} - 1.0 + \log \sigma_{p_i}^2  - \log \sigma_{q_i}^2 \right)$$
    """
    def __init__(self, q=None, p=None, q_repeats=1, p_repeats=1, **kwargs):
        if q is not None and p is not None:
            self.state = 1
            incoming = (q, p)
        elif q is not None:
            self.state = 2
            incoming = q
        elif p is not None:
            self.state = 3
            incoming = p
        else:
            raise ValueError("Both q and p are None")
        super(GaussianKL, self).__init__(incoming,
                                         x_repeats=q_repeats,
                                         y_repeats=p_repeats,
                                         max_inputs=2, **kwargs)

    def extract_q_p(self, inputs, expand=False, get_q_p=False):
        assert len(self.input_shapes) == len(inputs)
        if self.state == 1:
            q, p = inputs
            n = T.int_div(q.shape[1], 2)
            q_mu = q[:, :n]
            q_sigma = q[:, n:]
            p_mu = p[:, :n]
            p_sigma = p[:, n:]
        elif self.state == 2:
            q, = inputs
            p = None
            n = T.int_div(q.shape[1], 2)
            q_mu = q[:, :n]
            q_sigma = q[:, n:]
            p_mu = 0
            p_sigma = 1
        elif self.state == 3:
            p, = inputs
            q = None
            n = T.int_div(p.shape[1], 2)
            p_mu = p[:, :n]
            p_sigma = p[:, n:]
            q_mu = 0
            q_sigma = 1
        else:
            raise ValueError("Unreachable state:", self.state)
        if expand:
            if self.state == 1 or self.state == 2:
                q_mu = utils.expand_variable(q_mu, self.repeats[0])
                q_sigma = utils.expand_variable(q_sigma, self.repeats[0])
            if self.state == 1 or self.state == 3:
                p_mu = utils.expand_variable(p_mu, self.repeats[1])
                p_sigma = utils.expand_variable(p_sigma, self.repeats[1])
        if get_q_p:
            return q, p, q_mu, q_sigma, p_mu, p_sigma
        else:
            return q_mu, q_sigma, p_mu, p_sigma

    def get_outputs_for(self, inputs, **kwargs):
        q_mu, q_sigma, p_mu, p_sigma = self.extract_q_p(inputs, True)
        # Trace
        kl = T.sqr(q_sigma)
        # Means difference
        kl += T.sqr(q_mu - p_mu)
        # Both times inverse of sigma p
        kl /= T.sqr(p_sigma)
        # Minus 1
        kl -= T.constant(1)
        # Log determinant of p_sigma
        kl += 2 * T.log(p_sigma)
        # Log determinant of q_sigma
        kl -= 2 * T.log(q_sigma)
        # Flatten
        kl = T.flatten(kl, outdim=2)
        # Calculate per data point
        kl = 0.5 * T.sum(kl, axis=1)
        return kl,

    def skfgn(self, optimizer, inputs, outputs, curvature, kronecker_inversion):
        assert len(curvature) == 1
        curvature = curvature[0]
        g_q, g_p = None, None
        if optimizer.variant == "skfgn-rp":
            q_mu, q_sigma, p_mu, p_sigma = self.extract_q_p(inputs, True)
            # Samples
            if self.state == 1:
                v_q_mu, v_q_sigma, v_p_mu, v_p_sigma = optimizer\
                    .random_sampler((q_mu.shape, q_sigma.shape, p_mu.shape, p_sigma.shape))
            elif self.state == 2:
                v_q_mu, v_q_sigma = optimizer.random_sampler((q_mu.shape, q_sigma.shape))
                v_p_mu, v_p_sigma = None, None
            elif self.state == 3:
                v_p_mu, v_p_sigma = optimizer.random_sampler((p_mu.shape, p_sigma.shape))
                v_q_mu, v_q_sigma = None, None
            else:
                raise NotImplementedError
            # Calculate Gauss-Newton matrices
            if self.state == 1 or self.state == 2:
                # Q
                g_q_mu = v_q_mu / p_sigma
                g_q_mu = utils.collapse_variable(g_q_mu, self.repeats[0])
                g_q_sigma = v_q_sigma * T.sqrt(T.inv(T.sqr(p_sigma)) + T.inv(T.sqr(q_sigma)))
                g_q_sigma = utils.collapse_variable(g_q_sigma, self.repeats[0])
                g_q = T.concatenate((g_q_mu, g_q_sigma), axis=1)
                g_q *= T.sqrt(curvature)
            if self.state == 1 or self.state == 3:
                # P
                g_p_mu = v_p_mu / T.sqr(p_sigma)
                g_p_mu = utils.collapse_variable(g_p_mu, self.repeats[1])
                g_p_sigma = 2 * v_p_sigma / T.sqr(p_sigma)
                g_p_sigma = utils.collapse_variable(g_p_sigma, self.repeats[1])
                g_p = T.concatenate((g_p_mu, g_p_sigma), axis=1)
                g_p *= T.sqrt(curvature)
        elif optimizer.variant == "skfgn-fisher" or optimizer.variant == "kfac*":
            q_mu, q_sigma, p_mu, p_sigma = self.extract_q_p(inputs, False)
            # Samples
            if self.state == 1:
                fake_q, fake_p = normal((q_mu.shape, p_mu.shape))
            elif self.state == 2:
                fake_q = normal(q_mu.shape)
                fake_p = None
            elif self.state == 3:
                fake_p = normal(p_mu.shape)
                fake_q = None
            else:
                raise NotImplementedError
            # Calculate gradients with from the samples
            if self.state == 1 or self.state == 2:
                # Q
                d_q_mu = fake_q / q_sigma
                d_q_sigma = (fake_q - 1) / q_sigma
                d_q = T.concatenate((d_q_mu, d_q_sigma), axis=1)
                g_q = d_q * T.sqrt(curvature)
            if self.state == 1 or self.state == 3:
                # P
                d_p_mu = fake_q / q_sigma
                d_p_sigma = (fake_p - 1) / p_sigma
                d_p = T.concatenate((d_p_mu, d_p_sigma), axis=1)
                g_p = d_p * T.sqrt(curvature)
        # elif optimizer.variant == "skfgn-i":
        #     raise NotImplementedError
        elif optimizer.variant == "kfra":
            q_mu, q_sigma, p_mu, p_sigma = self.extract_q_p(inputs, True)
            if self.state == 1 or self.state == 2:
                # Q
                if self.state == 1:
                    g_q_mu = T.mean(T.inv(T.sqr(p_sigma)), axis=0)
                else:
                    g_q_mu = T.ones(q_sigma.shape[1:])
                g_q_sigma = T.mean(T.inv(T.sqr(p_sigma)) + T.inv(T.sqr(q_sigma)), axis=0)
                g_q = T.diag(T.concatenate((g_q_mu, g_q_sigma), axis=0))
                g_q *= curvature
            if self.state == 1 or self.state == 3:
                # P
                g_p_mu = T.mean(T.inv(p_sigma), axis=0)
                g_p_sigma = 2 * g_p_mu
                g_p = T.diag(T.concatenate((g_p_mu, g_p_sigma), axis=0))
                g_p *= curvature
        else:
            raise ValueError("Unreachable!")
        if self.state == 1:
            return g_q, g_p
        elif self.state == 2:
            return g_q,
        elif self.state == 3:
            return g_p,
        else:
            raise NotImplementedError

    def gauss_newton_product(self, inputs_map, outputs_map, params, variant, v1, v2=None):
        q, p, q_mu, q_sigma, p_mu, p_sigma = self.extract_q_p(inputs_map[self], True, True)
        if self.state == 1 or self.state == 2:
            Jv1_q = T.Rop(q, params, v1)
            Jv1_q = utils.expand_variable(Jv1_q, self.repeats[0])
            if v2 is not None:
                Jv2_q = T.Rop(q, params, v2)
                Jv2_q = utils.expand_variable(Jv2_q, self.repeats[0])
            else:
                Jv2_q = Jv1_q
            # The Gauss-Newton for the last layer
            if variant == "skfgn-fisher" or variant == "kfac*":
                g_q_mu = T.inv(T.sqr(q_sigma))
                g_q_sigma = 2 * g_q_mu
            elif self.state == 1:
                g_q_mu = T.inv(T.sqr(p_sigma))
                g_q_sigma = T.inv(T.sqr(p_sigma)) + T.inv(T.sqr(q_sigma))
            else:
                g_q_mu = T.ones_like(q_mu)
                g_q_sigma = 1 + T.inv(T.sqr(q_sigma))
            g_q = T.concatenate((g_q_mu, g_q_sigma), axis=1)
            v1Jv2_q = T.sum(Jv1_q * g_q * Jv2_q, axis=1)
            v1Jv2_q = T.mean(v1Jv2_q)
        else:
            v1Jv2_q = 0
        if self.state == 1 or self.state == 3:
            Jv1_p = T.Rop(p, params, v1)
            Jv1_p = utils.expand_variable(Jv1_p, self.repeats[1])
            if v2 is not None:
                Jv2_p = T.Rop(p, params, v2)
                Jv2_p = utils.expand_variable(Jv2_p, self.repeats[1])
            else:
                Jv2_p = Jv1_p
            # The Gauss-Newton for the last layer
            g_p_mu = T.inv(T.sqr(p_sigma))
            g_p_sigma = 2 * g_p_mu
            g_p = T.concatenate((g_p_mu, g_p_sigma), axis=1)
            v1Jv2_p = T.sum(Jv1_p * g_p * Jv2_p, axis=1)
            v1Jv2_p = T.mean(v1Jv2_p)
        else:
            v1Jv2_p = 0
        return v1Jv2_q + v1Jv2_p
