"""
A module with a package-wide random number generator,
used for weight initialization and seeding noise layers.
This can be replaced by a :class:`numpy.random.RandomState` instance with a
particular seed to facilitate reproducibility.
"""

from functools import wraps
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.gradient import zero_grad
from .utils import th_fx, linear_solve

_rng = np.random


def get_rng():
    """
    Get the package-level random number generator.

    Returns
    -------
    :class:`numpy.random.RandomState` instance
        The :class:`numpy.random.RandomState` instance passed to the most
        recent call of :func:`set_rng`, or ``numpy.random`` if :func:`set_rng`
        has never been called.
    """
    return _rng


def set_rng(new_rng):
    """
    Set the package-level random number generator.

    Parameters
    ----------
    new_rng : ``numpy.random`` or a :class:`numpy.random.RandomState` instance
        The random number generator to use.
    """
    global _rng
    _rng = new_rng


def get_symbolic_shape_ndim(shape):
    """
    Returns the number of dimension of a result of `theano.tensor.shape`

    Parameters
    ----------
    shape : Theano expression
        This needs to be the result of `theano.tensor.shape`

    Returns
    ----------
    The number of dimensions of the shape.
    """
    try:
        _, = shape
        return 1
    except ValueError:
        pass
    try:
        _, _ = shape
        return 2
    except ValueError:
        pass
    try:
        _, _, _ = shape
        return 3
    except ValueError:
        pass
    try:
        _, _, _, _ = shape
        return 4
    except ValueError:
        pass
    raise ValueError("Something went wrong")


def multi_sampler(sampler):
    """
    Wraps a standard sampling method to be able to take as input
    multiple shapes.

    Parameters
    ----------
    sampler : The sampling method

    Returns
    ----------
    A sampling method equivalent to the input, but allowing to pass
    multiple shapes
    """
    @wraps(sampler)
    def sample(shapes, *args, **kwargs):
        # Shapes can be either a single shape or a list of shapes
        if isinstance(shapes[0], (T.TensorVariable, theano.tensor.sharedvar.TensorSharedVariable)):
            # First entry is a symbolic expression
            if shapes[0].ndim == 0:
                # Single shape
                return sampler(shapes, *args, **kwargs)
            else:
                # Multiple shapes
                ndim = get_symbolic_shape_ndim(shapes[0])
        elif isinstance(shapes[0], int):
            # Single shape
            return sampler(shapes, *args, **kwargs)
        elif isinstance(shapes[0], (list, tuple, np.ndarray)):
            # Multiple shapes
            ndim = len(shapes[0])
        else:
            raise ValueError("Incorrect shapes input - {}".format(shapes))
        # Code for multiple shapes
        for s in shapes:
            if isinstance(s, (list, tuple, np.ndarray)):
                assert ndim == len(s)
            else:
                assert ndim == get_symbolic_shape_ndim(s)
        n = sum(s[0] for s in shapes)
        if ndim == 1:
            all_shape = (n, )
        elif ndim == 2:
            all_shape = (n, shapes[0][1])
        elif ndim == 3:
            all_shape = (n, shapes[0][1], shapes[0][2])
        else:
            all_shape = (n, shapes[0][1], shapes[0][2], shapes[0][3])
        samples = sampler(all_shape, *args, **kwargs)
        n = 0
        split = []
        for s in shapes:
            split.append(samples[n:n + s[0]])
            n += s[0]
        return split
    return sample


@multi_sampler
def uniform(shape, min=0, max=1, dtype=None, random=None):
    """
    Draws a symbolic sample from the uniform hypercube.

    Parameters
    ----------
    shape: sequence of int or theano shape, can be symbolic
        Desired shape of the sample
    min: float or array, optional, can be symbolic
        Lower bound of the uniform distribution. Defaults to 0.
    max: float or array, optional, can be symbolic
        Upper bound of the uniform distribution. Defaults to 1.
    dtype: numpy data-type, optional
        The desired dtype of the sample. Defaults to ``floatX``.
    random: Theano RandomStreams or None
        An optional RandomStreams to be used for sampling.

    Returns
    -------
        A Theano variable that contains a sample from the uniform hypercube.
    """
    dtype = dtype or theano.config.floatX
    random = random or RandomStreams(get_rng().randint(1, 2147462579))
    samples = random.uniform(shape, dtype=dtype)
    samples = zero_grad(samples)
    if min != 0 or max != 1:
        samples *= (max - min)
        samples += min
    return samples


@multi_sampler
def normal(shape, mean=0, std=1, dtype=None, random=None):
    """
    Draws a symbolic sample from a Gaussian distribution.

    Parameters
    ---------
    shape: sequence of int or theano shape, can be symbolic
        Shape of the sample.
    mean: float or array, can be symbolic
        Mean of the sample.
    std: float or array, can be symbolic
        Standard deviation of the sample.
    dtype: numpy data-type, optional
        The desired dtype of the sample. Defaults to ``floatX``.
    random: Theano RandomStreams or None
        An optional RandomStreams to be used for sampling.

    Returns
    -------
        A Theano variable that contains a sample from the Gaussian
        distribution.
    """
    dtype = dtype or theano.config.floatX
    random = random or RandomStreams(get_rng().randint(1, 2147462579))
    samples = random.normal(shape, dtype=dtype)
    samples = zero_grad(samples)
    if std != 1:
        samples *= std
    if mean != 0:
        samples += mean
    return samples


@multi_sampler
def multinomial(shape, n=1, pvals=None, dtype=None, random=None):
    """
     Draws a symbolic sample from a Multinomial distribution.

     Parameters
     ----------
     shape: sequence of int or theano shape, can be symbolic
         Shape of the sample.
     n: int
         Sample `n` times from a multinomial distribution defined by pvals.
     pvals: vector or None
         The distribution to sample from. If None samples uniformly.
     dtype: numpy data-type, optional
         The desired dtype of the sample. Defaults to ``floatX``.
     random: Theano RandomStreams or None
         An optional RandomStreams to be used for sampling.

     Returns
     -------
         A Theano variable that contains a sample from a binary distribution.
     """
    random = random or RandomStreams(get_rng().randint(1, 2147462579))
    pvals = pvals or T.ones(shape) / th_fx(shape[1])
    samples = random.multinomial(pvals=pvals, n=n)
    samples = samples if dtype is None else T.cast(samples, dtype)
    return samples


@multi_sampler
def binary(shape, p=0.5, v0=0, v1=1, dtype=None, random=None):
    """
    Draws a symbolic sample from a Bernoulli distribution.

    Parameters
    ----------
    shape: sequence of int or theano shape, can be symbolic
        Shape of the sample.
    p: float, optional, 0 <= p <= 1
        Probability of sampling ``v0``. Defaults to 0.5
    v0: number, optional
        Sample value 0. Defaults to 0.
    v1: number, optional
        Sample value 1. Defaults to 1.
    dtype: numpy data-type, optional
        The desired dtype of the sample. Defaults to ``floatX``.
    random: Theano RandomStreams or None
        An optional RandomStreams to be used for sampling.
    
    Returns
    -------
        A Theano variable that contains a sample from a binary distribution.
    """
    dtype = dtype or theano.config.floatX
    if v0 < v1:
        v0, v1 = v1, v0
        p = 1 - p
    random = random or RandomStreams(get_rng().randint(1, 2147462579))
    samples = random.binomial(shape, p=p, dtype=dtype)
    samples = zero_grad(samples)
    if v0 != 0 or v1 != 1:
        samples *= v1 - v0
        samples += v0
    return samples


@multi_sampler
def uniform_ball(shape, radius=None, dtype=None, random=None):
    """
    Draws a sample from the uniform hypersphere.

    Parameters
    ----------
    shape: sequence of int or theano shape, can be symbolic
        Shape of the sample
    radius: float, optional
        Radius of the hypersphere. Defaults to sqrt(dim+2) in order to make the
        sample zero-mean and unit-variance.
    dtype: numpy data-dtype, optional
        The desired dtype of the sample. Defaults to ``floatX``.
    random: Theano RandomStreams or None
        An optional RandomStreams to be used for sampling.
    
    Returns
    -------
        A Theano variable that contains a sample from the uniform hypersphere.
    """
    dtype = dtype or theano.config.floatX
    dim = T.cast(shape[-1], dtype)
    radius = radius or T.sqrt(dim + 2)
    u = uniform(shape, dtype=dtype, random=random)
    x = normal(shape, dtype=dtype, random=random)
    distance = radius * u ** (1 / dim)
    norm = T.sqrt(T.sum(T.sqr(x), axis=-1, keepdims=True))
    return distance / norm * x


@multi_sampler
def matrix_normal(shape, M=None, A=None, B=None, dtype=None, mode="cov"):
    dtype = dtype or theano.config.floatX
    if len(shape) == 2:
        n, p = shape
    else:
        n, p = shape[1:]
    if M is None:
        M = T.zeros((n, p), dtype=dtype)
    if A is None:
        A = T.eye(n)
    if B is None:
        B = T.eye(p)

    X = normal(shape, dtype=dtype)
    if M is None and A is None and B is None:
        return X

    if len(shape) == 2:
        if mode == "cov":
            return M + A.dot(X).dot(B)
        elif mode == "prec":
            X1 = linear_solve(A, X, A_structure="upper_triangular")
            X2 = linear_solve(B, X1.T, A_structure="lower_triangular").T
            return M + X2
    else:
        if mode == "prec":
            raise NotImplemented
        M = M.dimshuffle(['x', 0, 1])
        return M + (X.dimshuffle([0, 2, 1]).dot(A.T)).dimshuffle([0, 2, 1]).dot(B)


# def gamma1(alpha, dtype=None, alpha_0=2.5, max_samples=20):
#     """
#     Samples from Gamma(alpha, 1)
#     The algorithm implemented is from:
#     https://www.jstor.org/stable/pdf/2347200.pdf?refreqid=excelsior:95c8b191b12ab247ecd96572fb6c10bd
#
#     :param alpha:
#     :param dtype:
#     :return:
#     """
#     alpha_flat = T.flatten(alpha, outdim=1)
#     shape = (alpha_flat.shape[0], max_samples)
#     a = alpha_flat - 1
#     b = (alpha_flat - T.inv(6 * alpha_flat)) / a
#     c = 2 / a
#     d = c + 2
#     f = T.sqrt(alpha_flat)
#     b = b.dimshuffle(0, 'x')
#     c = c.dimshuffle(0, 'x')
#     d = d.dimshuffle(0, 'x')
#     f = f.dimshuffle(0, 'x')
#     u1, u2 = uniform((shape, shape), dtype=dtype)
#
#     # GKM 1
#     w = b * u1 / u2
#     cond1 = T.le(c * u2 - d + w + T.inv(w), 0)
#     cond2 = T.lt(c * T.log(u2) - T.log(w) + w - 1, 0)
#     deliver_1 = T.or_(cond1, cond2)
#
#     # GKM 2
#     u3 = u1 + (1 - 1.86 * u2) / f
#     cond3 = T.and_(T.gt(u3, 0), T.lt(u3, 1))
#     deliver_2 = T.and_(deliver_1, cond3)
#
#     # GKM 3
#     deliver = T.switch(T.lt(alpha_flat, alpha_0).dimshuffle(0, 'x'),
#                        deliver_1, deliver_2)
#     index = T.argmax(deliver, axis=1)
#     x = a * w[T.arange(w.shape[0]), index]
#     return T.reshape(x, alpha.shape)


def gamma(alpha, dtype=None, max_samples=5):
    """
    Samples from Gamma(alpha, 1)
    The algorithm implemented is from:
    http://dl.acm.org/citation.cfm?id=358414

    """
    alpha_flat = T.flatten(alpha, outdim=1)
    shape = (alpha_flat.shape[0], max_samples)
    d = alpha_flat - T.constant(1.0 / 3.0)
    c = T.inv(T.sqrt(9 * d))
    x = normal(shape, dtype=dtype)
    v = (1 + c.dimshuffle(0, 'x') * x) ** 3
    u = uniform(shape, dtype=dtype)
    cond1 = T.lt(u, 1 - 0.0331 * (x ** 4))
    cond2 = T.lt(T.log(u), 0.5 * (x ** 2) + d.dimshuffle(0, 'x') * (1 - v + T.log(v)))
    cond3 = T.or_(cond1, cond2)
    deliver = T.and_(T.gt(v, 0), cond3)
    index = T.argmax(deliver, axis=1)
    y = d * v[T.arange(alpha_flat.shape[0]), index]
    return T.reshape(y, alpha.shape)


def beta(alpha, beta, dtype=None):
    """
    Samples from a beta distribution with parameters alpha and beta of the same shape.
    """
    n = alpha.shape[0]
    g = gamma(T.concatenate((alpha, beta), axis=0), dtype=dtype)
    gamma_a = g[:n]
    gamma_b = g[n:]
    return gamma_a / (gamma_a + gamma_b)
