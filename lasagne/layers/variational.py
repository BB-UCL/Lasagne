import theano
import theano.tensor as T
from theano.gradient import disconnected_grad
import numpy as np

from .base import Layer
from .special import NonlinearityLayer
from .losses import SKFGNLossLayer
from .. import nonlinearities
from .. import utils
from ..random import normal


__all__ = [
    "GaussianParameters",
    "GaussianSampler",
    "GaussianKL",
    "GaussianLikelihood",
    "GaussianEntropy",
    "IWAEBound"
]


# class GaussianParameters(Layer):
#     """
#     A layer that will transform a concatenated mean and pre-sigma to
#     a concatenated mean and sigma.
#     The transformation is only on pre-sigma via some nonlinear function
#     which should always output positive values.
#
#     Parameters
#     ----------
#     incoming : a :class:`Layer` instance or a tuple or a list
#         The layer feeding into this layer, or the expected input shape
#
#     nonlinearity : callable
#         The function to apply to sigma. By default it is
#         `lasagne.nonlinearities.softplus`.
#
#     eps : a float
#         A conditioning number to ensure the final sigma is not 0.
#     """
#     def __init__(self, incoming, nonlinearity=nonlinearities.softplus,
#                  eps=1e-6, **kwargs):
#         super(GaussianParameters, self).__init__(incoming, **kwargs)
#         self.nonlinearity = nonlinearity
#         self.eps = eps
#
#     @staticmethod
#     def get_raw(distribution):
#         d = T.int_div(distribution.shape[1], 2)
#         mu = distribution[:, :d]
#         pre_sigma = distribution[:, d:]
#         return mu, pre_sigma
#
#     def get_outputs_for(self, inputs, **kwargs):
#         assert len(inputs) == 1
#         mu, pre_sigma = GaussianParameters.get_raw(inputs[0])
#         sigma = self.nonlinearity(pre_sigma) + self.eps
#         return T.concatenate((mu, sigma), axis=1),
#
#     def curvature_propagation(self, optimizer, inputs, outputs, curvature, make_matrix):
#         assert len(inputs) == 1
#         assert len(curvature) == 1
#         assert len(outputs) == 1
#         if optimizer.variant == "kfra":
#             _, pre_sigma = GaussianParameters.get_raw(inputs[0])
#             n = utils.th_fx(pre_sigma.shape[0])
#             sigma = self.nonlinearity(pre_sigma) + self.eps
#             a = T.Lop(sigma, pre_sigma, T.ones_like(sigma))
#             a = T.concatenate((T.ones_like(a), a), axis=1)
#             return curvature[0] * T.dot(a.T, a) / n,
#         else:
#             return T.Lop(outputs[0], inputs[0], curvature[0]),


def GaussianParameters(incoming, **kwargs):
    return NonlinearityLayer(incoming,
                             nonlinearity=nonlinearities.gaussian_parametrization,
                             **kwargs)


class GaussianSampler(Layer):
    """
    A layer that will sample for an isotropic Gaussian distribution.
    It can either take a single input, which contains the concatenated
    mean and standard deviation as returned from `GaussianParameters`
    or have to incoming layers one providing the mean and one the standard
    deviation.

    The result will be flattened on to axis 0. For instance if you input
    a mean of size (N, D), the layer will sample (N, K, D) and reshape it
    to (NK, D).

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple or list
        The layer/s feeding into this layer, or the expected input shape/s

    num_samples : int (default: 1)
        How many samples to take per data point.

    output_ll: bool (default: False)
        Whether to output the log likelihood of the samples under the
        parameters of the Gaussian distribution.
    """
    def __init__(self, incoming,
                 num_samples=1,
                 output_epsilon=False,
                 **kwargs):
        super(GaussianSampler, self).__init__(incoming, **kwargs)
        self.num_samples = num_samples
        self.output_epsilon = output_epsilon

    def get_output_shapes_for(self, input_shapes):
        shape = input_shapes[0]
        assert shape[1] % 2 == 0
        out_shape = (shape[0] * self.num_samples if shape[0] is not None else None, shape[1] // 2)
        if self.output_epsilon:
            return out_shape, (shape[0], self.num_samples, shape[1] // 2)
        else:
            return out_shape,

    def get_outputs_for(self, inputs, deterministic=False, **kwargs):
        mu, sigma = utils.split_half(inputs[0])
        n, d = mu.shape
        if deterministic:
            return mu,

        epsilon = normal((n, self.num_samples, d))
        samples = mu.dimshuffle(0, 'x', 1) + sigma.dimshuffle(0, 'x', 1) * epsilon
        samples = T.reshape(samples, (n * self.num_samples, d))
        if self.output_epsilon:
            return samples, epsilon
        else:
            return samples,

    def curvature_propagation(self, optimizer, inputs, outputs, curvature, make_matrix):
        if not self.output_epsilon:
            assert len(inputs) == 1
            assert len(curvature) == 1
            assert len(outputs) == 1
        else:
            assert len(inputs) == 1
            assert len(curvature) == 2
            assert len(outputs) == 2

        if optimizer.variant == "kfra":
            samples = outputs[0].owner.inputs[0]
            if not isinstance(samples.owner.op, theano.tensor.elemwise.Elemwise):
                samples = outputs[0].owner.inputs[1]
            if not isinstance(samples.owner.op, theano.tensor.elemwise.Elemwise):
                raise ValueError("Something is wrong with getting samples.")
            sigma_eps = samples.owner.inputs[1]
            if not isinstance(sigma_eps.owner.op, theano.tensor.elemwise.Elemwise):
                sigma_eps = samples.owner.inputs[0]
            if not isinstance(sigma_eps.owner.op, theano.tensor.elemwise.Elemwise):
                raise ValueError("Something is wrong with getting sigma_eps.")
            epsilon = sigma_eps.owner.inputs[1]
            if not isinstance(epsilon.owner.op, theano.gradient.ZeroGrad):
                epsilon = sigma_eps.owner.inputs[0]
            if not isinstance(epsilon.owner.op, theano.gradient.ZeroGrad):
                raise ValueError("Something is wrong with getting epsilon.")
            flat_epsilon = T.flatten(epsilon, ndim=2)
            nk = utils.th_fx(epsilon.shape[0])
            ge = T.tile(curvature[0], (2, 2))
            eps_extended = T.concatenate((T.ones_like(flat_epsilon), flat_epsilon), axis=1)
            return ge * T.dot(eps_extended.T, eps_extended) / nk,
        elif optimizer.variant == "kfac*":
            return T.constant(0),
        else:
            return T.Lop(outputs[0], inputs[0], curvature[0]),


class GaussianKL(SKFGNLossLayer):
    """
    A layer that calculates the KL(q||p) for isotropic Gaussian distributions.
    If one of the distributions is omitted it is assumed to be the standard
    normal.

    The inputs are assumed to represent the concatenation of a mean and a
    standard deviation as parametrized by `GaussianParameters` layer.

    Formally:
    KL(q||p) = \frac{1}{2} \sum_i \left( \frac{\sigma_{q_i}^2}{\sigma_{p_i}^2} +
    \frac{(\mu_{p_i} - \mu_{q_i})^2}{\sigma_{p_i}^2} - 1.0 + \log \sigma_{p_i}^2  - \log \sigma_{q_i}^2 \right)

    Parameters
    ----------
    q : a :class:`Layer` instance or a tuple or a list
        The layer or the expected input shape for `q`.

    p : a :class:`Layer` instance or a tuple or a list
        The layer or the expected input shape for `q`.

    q_repeats: int (default: 1)
        If `q` has to be repeated some number of times.

    p_repeats: int (default: 1)
        If `p` has to be repeated some number of times.
    """
    def __init__(self, q=None, p=None, q_repeats=1, p_repeats=1, **kwargs):
        if q is not None and p is not None:
            self.state = 1
            incoming = (q, p)
            repeats = (q_repeats, p_repeats)
        elif q is not None:
            self.state = 2
            incoming = q
            repeats = (q_repeats, )
        elif p is not None:
            self.state = 3
            incoming = p
            repeats = (p_repeats, )
        else:
            raise ValueError("Both q and p are None")
        super(GaussianKL, self).__init__(incoming, repeats,
                                         max_inputs=2, **kwargs)

    def extract_q_p(self, inputs, expand=False, get_q_p=False):
        assert len(self.input_shapes) == len(inputs)
        if self.state == 1:
            q, p = inputs
            q_mu, q_sigma = utils.split_half(q)
            p_mu, p_sigma = utils.split_half(p)
        elif self.state == 2:
            q, p = inputs[0], None
            q_mu, q_sigma = utils.split_half(q)
            p_mu = T.constant(0)
            p_sigma = T.constant(1)
        elif self.state == 3:
            q, p = None, inputs[0]
            q_mu = T.constant(0)
            q_sigma = T.constant(1)
            p_mu, p_sigma = utils.split_half(p)
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
        kl = T.flatten(kl, ndim=2)
        # Calculate per data point
        kl = 0.5 * T.sum(kl, axis=1)
        return kl,

    def curvature_propagation(self, optimizer, inputs, outputs, curvature, make_matrix):
        assert len(curvature) == 1
        weight = T.sqrt(curvature[0])
        g_q, g_p = None, None
        if optimizer.variant == "skfgn-rp":
            q_mu, q_sigma, p_mu, p_sigma = self.extract_q_p(inputs, True)
            # Samples
            if self.state == 1:
                v_q_mu, v_q_sigma, v_p_mu, v_p_sigma = optimizer \
                    .random_sampler((q_mu.shape, q_sigma.shape, p_mu.shape, p_sigma.shape))
            elif self.state == 2:
                v_q_mu, v_q_sigma = optimizer.random_sampler((q_mu.shape, q_sigma.shape))
                v_p_mu, v_p_sigma = None, None
            elif self.state == 3:
                v_p_mu, v_p_sigma = optimizer.random_sampler((p_mu.shape, p_sigma.shape))
                v_q_mu, v_q_sigma = None, None
            else:
                # TODO
                raise NotImplementedError
            # Calculate Gauss-Newton matrices
            if self.state == 1 or self.state == 2:
                # Q
                g_q_mu = v_q_mu / p_sigma
                g_q_mu = utils.collapse_variable(g_q_mu, self.repeats[0])
                g_q_sigma = v_q_sigma * T.sqrt(T.inv(T.sqr(p_sigma)) + T.inv(T.sqr(q_sigma)))
                g_q_sigma = utils.collapse_variable(g_q_sigma, self.repeats[0])
                g_q = T.concatenate((g_q_mu, g_q_sigma), axis=1)
                g_q *= weight
            if self.state == 1 or self.state == 3:
                # P
                g_p_mu = v_p_mu / p_sigma
                g_p_mu = utils.collapse_variable(g_p_mu, self.repeats[1])
                g_p_sigma = T.sqrt(2) * v_p_sigma / p_sigma
                g_p_sigma = utils.collapse_variable(g_p_sigma, self.repeats[1])
                g_p = T.concatenate((g_p_mu, g_p_sigma), axis=1)
                g_p *= weight
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
                # TODO
                raise NotImplementedError
            # Calculate gradients with from the samples
            if self.state == 1 or self.state == 2:
                # Q
                d_q_mu = fake_q / q_sigma
                d_q_sigma = (1 - T.sqr(fake_q)) / q_sigma
                d_q = T.concatenate((d_q_mu, d_q_sigma), axis=1)
                g_q = d_q * weight
            if self.state == 1 or self.state == 3:
                # P
                d_p_mu = fake_p / p_sigma
                d_p_sigma = (1 - T.sqr(fake_p)) / p_sigma
                d_p = T.concatenate((d_p_mu, d_p_sigma), axis=1)
                g_p = d_p * weight
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
                g_q *= weight**2
            if self.state == 1 or self.state == 3:
                # P
                g_p_mu = T.mean(T.inv(p_sigma), axis=0)
                g_p_sigma = 2 * g_p_mu
                g_p = T.diag(T.concatenate((g_p_mu, g_p_sigma), axis=0))
                g_p *= weight**2
        else:
            raise ValueError("Unreachable!")
        if self.state == 1:
            return g_q, g_p
        elif self.state == 2:
            return g_q,
        elif self.state == 3:
            return g_p,
        else:
            # TODO
            raise NotImplementedError

    def gauss_newton_product(self, inputs_map, outputs_map, params, variant, v1, v2=None):
        q, p, q_mu, q_sigma, p_mu, p_sigma = self.extract_q_p(inputs_map[self], True, True)
        if self.state == 1 or self.state == 2:
            Jv1_q = utils.Rop(q, params, v1)
            Jv1_q = utils.expand_variable(Jv1_q, self.repeats[0])
            if v2 is not None:
                Jv2_q = utils.Rop(q, params, v2)
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
            Jv1_p = utils.Rop(p, params, v1)
            Jv1_p = utils.expand_variable(Jv1_p, self.repeats[1])
            if v2 is not None:
                Jv2_p = utils.Rop(p, params, v2)
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


class GaussianLikelihood(SKFGNLossLayer):
    """
    A layer that calculates the log likelihood of 'x' under the isotropic
    Gaussian distribution `p`. If `p` is None it is assumed to be the standard
    normal distribution.

    The distribution `p` should be represented as the concatenation of a mean
    and a standard deviation as parametrized by `GaussianParameters` layer.
    If `p` is None than it is assumed to be a Standard Gaussian.

    Formally:
        - 0.5 log(2 pi) - log(sigma) - 0.5 (x - mu)^2/sigma^2

    Parameters
    ----------
    x : a :class:`Layer` instance or a tuple or a list
        The layer or the expected input shape for `x`.

    p : a :class:`Layer` instance or a tuple or a list
        The layer or the expected input shape for `p`.

    num_samples: int (default: 1)
        The number of samples that were taken in `x`.

    path_derivative: bool (default: False)
        If True when calculating the log likelihood it will be in such form
        that taking gradients with respect to it will give you the
        PathDerivative rather than the usual estimator.
    """
    def __init__(self, x, p=None, x_repetas=1, p_repeats=1,
                 unit_variance=False, path_derivative=False,
                 *args, **kwargs):
        if p is None:
            incoming = (x, )
            repeats = (x_repetas, )
            mn = 1
        else:
            incoming = (x, p)
            repeats = (x_repetas, p_repeats)
            mn = 2
        super(GaussianLikelihood, self).__init__(incoming, max_inputs=mn,
                                                 repeats=repeats,
                                                 *args, **kwargs)
        self.unit_variance = unit_variance
        self.path_derivative = path_derivative

    def get_outputs_for(self, inputs, **kwargs):
        x = utils.expand_variable(inputs[0], self.repeats[0])
        x = T.flatten(x, ndim=2)
        if len(inputs) == 2:
            if self.unit_variance:
                mu = inputs[1]
                mu = utils.expand_variable(T.flatten(mu, ndim=2), self.repeats[1])
                sigma = T.constant(1)
            else:
                mu, sigma = utils.split_half(inputs[1])
                mu = utils.expand_variable(T.flatten(mu, ndim=2), self.repeats[1])
                sigma = utils.expand_variable(T.flatten(sigma, ndim=2), self.repeats[1])
            if self.path_derivative:
                mu = disconnected_grad(mu)
                sigma = disconnected_grad(sigma)
            log_p_x = - 0.5 * T.log(2 * np.pi)
            log_p_x += - T.log(sigma) - 0.5 * T.sqr((x - mu) / sigma)
        else:
            assert len(inputs) == 1
            log_p_x = - 0.5 * T.log(2 * np.pi) - 0.5 * T.sqr(x)
        return T.sum(log_p_x, axis=1),

    def curvature_propagation(self, optimizer, inputs, outputs, curvature, make_matrix):
        assert len(curvature) == 1
        weight = T.sqrt(curvature[0])
        x = inputs[0]
        if len(inputs) == 2:
            if self.unit_variance:
                mu = inputs[1]
                if optimizer.variant == "skfgn-rp":
                    cl_v, cl_mu = optimizer.random_sampler((x.shape, mu.shape))
                    cl_v *= weight
                    cl_mu *= weight
                    return cl_v, cl_mu
                elif optimizer.variant == "skfgn-fisher" or optimizer.variant == "kfac*":
                    fake_dx, fake_dmu = normal((x.shape, mu.shape))
                    fake_dx *= weight
                    fake_dmu *= weight
                    return fake_dx, fake_dmu
                elif optimizer.variant == "kfra":
                    if x.ndim == 1:
                        gn_x = weight**2
                        gn_mu = weight**2
                    elif x.ndim == 2:
                        x = T.flatten(x, ndim=2)
                        gn_x = T.eye(x.shape[1])
                        gn_x *= weight**2
                        mu = T.flatten(mu, ndim=2)
                        gn_mu = T.eye(mu.shape[1])
                        gn_mu *= weight**2
                    else:
                        raise ValueError("KFRA can not work on more than 2 dimensions.")
                    return gn_x, gn_mu
                else:
                    raise ValueError("Unreachable!")
            else:
                # TODO
                raise NotImplementedError
        else:
            assert len(inputs) == 1
            if optimizer.variant == "skfgn-rp":
                cl_v = optimizer.random_sampler(x.shape)
                cl_v *= weight
                return cl_v,
            elif optimizer.variant == "skfgn-fisher" or optimizer.variant == "kfac*":
                fake_dx = normal(x.shape)
                fake_dx *= weight
                return fake_dx,
            elif optimizer.variant == "kfra":
                if x.ndim == 1:
                    gn_x = weight**2
                elif x.ndim == 2:
                    x = T.flatten(x, ndim=2)
                    gn_x = T.eye(x.shape[1])
                    gn_x *= weight**2
                else:
                    raise ValueError("KFRA can not work on more than 2 dimensions.")
                return gn_x,
            else:
                raise ValueError("Unreachable!")

    def gauss_newton_product(self, inputs_map, outputs_map, params, variant, v1, v2=None):
        x = inputs_map[self][0]
        if len(self.input_shapes) == 2:
            if self.unit_variance:
                mu = inputs_map[self][1]
                jv_x = utils.gauss_newton_product(x, T.constant(1), params, v1, v2)
                jv_mu = utils.gauss_newton_product(mu, T.constant(1), params, v1, v2)
                return - jv_x - jv_mu
            else:
                # TODO
                raise NotImplementedError
        else:
            return utils.gauss_newton_product(x, T.constant(1), params, v1, v2)


class GaussianEntropy(SKFGNLossLayer):
    """
    A layer that calculates the entropy of an isotropic Gaussian distribution.

    The distribution `p` should be represented as the concatenation of a mean
    and a standard deviation as parametrized by `GaussianParameters` layer.

    Formally:
        0.5 * log(2 pi) + 0.5 + T.log(sigma)

    Parameters
    ----------
    p : a :class:`Layer` instance or a tuple or a list
        The layer or the expected input shape for `p`.
    """
    def __init__(self, p, *args, **kwargs):
        super(GaussianEntropy, self).__init__(p, *args, **kwargs)

    def get_outputs_for(self, inputs, **kwargs):
        assert len(inputs) == 1
        _, sigma = utils.split_half(inputs[0])
        entropy = T.log(T.constant(2 * np.pi)) / 2.0 + 0.5 + T.log(sigma)
        return T.sum(entropy, axis=1),

    def curvature_propagation(self, optimizer, inputs, outputs, curvature, make_matrix):
        # TODO
        raise NotImplementedError

    def gauss_newton_product(self, inputs_map, outputs_map, params, variant, v1, v2=None):
        # TODO
        raise NotImplementedError


class IWAEBound(SKFGNLossLayer):
    """
    A layer that calculates the IWAE bound.
    The inputs to the layer should be in the form of logits.
    The weights provide an easy way of controlling weighting of *THE LOGITS*.

    For standard IWAE bound you will have three inputs:
        log p(x|z)
        log p(z)
        log q(z|x)

    More inputs are allowed for things like Normalizing Flows.

    Given inputs x1, ..., xn the layer calculates with shapes (NK, D)
    the layer first reshapes them to (N, K, D) and then calculates:
        log mean(exp(x1 + x2 + x3), axis=1)
    but in a stable way.

    Parameters
    ----------
    incoming : a :class:`Layer` instances or a tuples or lists
        The layers or the expected input shapes

    weights: tuple or list or None
        Any scalar weighting which to apply to the logits.
    """
    def __init__(self, incoming, weights=None, num_samples=1,
                 **kwargs):
        super(IWAEBound, self).__init__(incoming, max_inputs=10, **kwargs)
        if weights is None:
            self.weights = [T.constant(1) for _ in self.input_shapes]
        else:
            self.weights = weights
        self.num_samples = num_samples

    def get_output_shapes_for(self, input_shapes):
        n = None
        for shape in input_shapes:
            if shape[0] is not None:
                if n is not None:
                    assert n == shape[0]
                else:
                    n = shape[0]
        return (n, ),

    def get_outputs_for(self, inputs, **kwargs):
        assert len(inputs) == len(self.weights)
        # Zero dimensional inputs, e.g. not dependent on the data
        s0 = 0
        # One dimensional inputs, e.g. dependent on the data
        s1 = 0
        for i, w in zip(inputs, self.weights):
            if i.ndim == 0:
                s0 += i * w
            elif i.ndim == 1:
                s1 += i * w
            else:
                raise ValueError("More than 1D input.")
        if self.num_samples == 1 or s1 == 0:
            return s0 + s1,
        s1 = T.reshape(s1, (-1, self.num_samples))
        m1 = disconnected_grad(T.max(s1, axis=1))
        s1_cond = s1 - m1.dimshuffle(0, 'x')
        return s0 + m1 + T.log(T.mean(T.exp(s1_cond), axis=1)),

    def curvature_propagation(self, optimizer, inputs, outputs, curvature, make_matrix):
        # TODO
        raise NotImplementedError

    def gauss_newton_product(self, inputs_map, outputs_map, params, variant, v1, v2=None):
        # TODO
        raise NotImplementedError
