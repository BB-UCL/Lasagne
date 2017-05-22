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
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.gradient import zero_grad

_rng = np.random

_mrng = None


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
    """Set the package-level random number generator.

    Parameters
    ----------
    new_rng : ``numpy.random`` or a :class:`numpy.random.RandomState` instance
        The random number generator to use.
    """
    global _rng
    _rng = new_rng


def get_mrng():
    global _mrng
    if _mrng is None:
        reset_mrng()
    return _mrng


def set_mrng(new_mrng):
    global _mrng
    _mrng = new_mrng


def reset_mrng():
    global _mrng
    _mrng = MRG_RandomStreams(seed=get_rng().randint(1000000))


def multi_sampler(sampler):
    @wraps(sampler)
    def sample(shapes, *args, **kwargs):
        if (isinstance(shapes[0], T.TensorVariable) and
                shapes[0].ndim == 0) or \
                (not isinstance(shapes[0], (list, tuple))):
            return sampler(shapes, *args, **kwargs)
        else:
            if isinstance(shapes[0], (list, tuple)):
                ndim = len(shapes[0])
            else:
                ndim = shapes[0].ndim
            for s in shapes:
                if isinstance(s, (list, tuple)):
                    assert ndim == len(s)
                else:
                    assert ndim == s.ndim
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
            return split
    return sample


@multi_sampler
def th_uniform(shape, min=0, max=1, dtype=None):
    dtype = dtype or theano.config.floatX
    samples = get_mrng().uniform(shape, dtype=dtype)
    samples = zero_grad(samples)
    if min != 0 or max != 1:
        samples *= (max - min) + min
    return samples


@multi_sampler
def th_normal(shape, mean=0, std=1, dtype=None):
    dtype = dtype or theano.config.floatX
    samples = get_mrng().normal(shape, dtype=dtype)
    samples = zero_grad(samples)
    if std != 1:
        samples *= std
    if mean != 0:
        samples += mean
    return samples


@multi_sampler
def th_binary(shape, p=0.5, v0=0, v1=1, dtype=None):
    dtype = dtype or theano.config.floatX
    if v0 < v1:
        temp = v0
        v0 = v1
        v1 = temp
        p = 1 - p
    samples = get_mrng().binomial(shape, p=p, dtype=dtype)
    samples = zero_grad(samples)
    if v0 != 0 or v1 != 1:
        samples *= v1 - v0
        samples += v0
    return samples


@multi_sampler
def th_uniform_ball(shape, radius=None, dtype=None):
    dim = T.cast(shape[-1], theano.config.floatX)
    radius = radius or T.sqrt(dim + 2)
    u = th_uniform(shape, dtype=dtype)
    x = th_normal(shape, dtype=dtype)
    distance = radius * u ** (1 / dim)
    norm = T.sqrt(T.sum(T.sqr(x), axis=-1, keepdims=True))
    return distance / norm * x
