import theano
import theano.tensor as T

from .base import Layer
from ..nonlinearities import softplus
from ..utils import th_fx
from ..random import th_normal


class GaussianParametersLayer(Layer):
    def __init__(self, incoming, nonlinearity=softplus, eps=1e-6, **kwargs):
        super(GaussianParametersLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = nonlinearity
        self.eps = eps

    @staticmethod
    def get_raw(inputs):
        x = inputs[0]
        d = T.int_div(x.shape[1], 2)
        mu = x[:, :d]
        pre_sigma = x[:, d:]
        return mu, pre_sigma

    def get_outputs_for(self, inputs, **kwargs):
        assert len(inputs) == 1
        mu, pre_sigma = GaussianParametersLayer.get_raw(inputs)
        sigma = self.nonlinearity(pre_sigma) + self.eps
        return T.concatenate((mu, sigma), axis=1),

    def skfgn(self, optimizer, inputs, outputs, curvature, kronecker_inversion):
        assert len(inputs) == 1
        assert len(curvature) == 1
        assert len(outputs) == 1

        if optimizer.variant == "kfra":
            _, pre_sigma = GaussianParametersLayer.get_raw(inputs)
            n = th_fx(pre_sigma.shape[0])
            sigma = self.nonlinearity(pre_sigma) + self.eps
            a = T.Lop(sigma, pre_sigma, T.ones_like(sigma))
            a = T.concatenate((T.ones_like(a), a), axis=1)
            return curvature[0] * T.dot(a.T, a) / n,
        else:
            return T.Lop(outputs[0], inputs[0], curvature[0]),


class GaussianSampleLayer(Layer):
    def __init__(self, incoming,
                 num_samples=1,
                 **kwargs):
        super(GaussianSampleLayer, self).__init__(incoming, **kwargs)
        self.num_samples = num_samples

    def get_output_shapes_for(self, input_shapes):
        shape = input_shapes[0]
        assert shape[1] % 2 == 0
        shape = (shape[0] * self.num_samples if shape[0] is not None else None, shape[1] // 2)
        return shape,

    def get_outputs_for(self, inputs, **kwargs):
        x = inputs[0]
        n = x.shape[0]
        d = T.int_div(x.shape[1], 2)
        mu = x[:, :d]
        sigma = x[:, d:]
        epsilon = th_normal((n, self.num_samples, d))
        samples = mu.dimshuffle(0, 'x', 1) + sigma.dimshuffle(0, 'x', 1) * epsilon
        return T.reshape(samples, (n * self.num_samples, d)),

    def skfgn(self, optimizer, inputs, outputs, curvature, kronecker_inversion):
        assert len(inputs) == 1
        assert len(curvature) == 1
        assert len(outputs) == 1

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
            d = epsilon.shape[-1]
            epsilon = T.reshape(epsilon, (-1, d))
            nk = th_fx(epsilon.shape[0])
            ge = T.tile(curvature[0], (2, 2))
            eps_extended = T.concatenate((T.ones_like(epsilon), epsilon), axis=1)
            return ge * T.dot(eps_extended.T, eps_extended) / nk,
        elif optimizer.variant == "kfac*":
            return T.constant(0),
        else:
            return T.Lop(outputs[0], inputs[0], curvature[0]),
