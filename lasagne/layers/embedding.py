import numpy as np
import theano.tensor as T

from .. import init
from .base import Layer


__all__ = [
    "EmbeddingLayer"
]


class EmbeddingLayer(Layer):
    """
    lasagne.layers.EmbeddingLayer(incoming, input_size, output_size,
    W=lasagne.init.Normal(), **kwargs)

    A layer for word embeddings. The input should be an integer type
    Tensor variable.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.

    input_size: int
        The Number of different embeddings. The last embedding will have index
        input_size - 1.

    output_size : int
        The size of each embedding.

    W : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the embedding matrix.
        This should be a matrix with shape ``(input_size, output_size)``.
        See :func:`lasagne.utils.create_param` for more information.

    Examples
    --------
    >>> from lasagne.layers import EmbeddingLayer, InputLayer, get_output
    >>> import theano
    >>> x = T.imatrix()
    >>> l_in = InputLayer((3, ))
    >>> W = np.arange(3*5).reshape((3, 5)).astype('float32')
    >>> l1 = EmbeddingLayer(l_in, input_size=3, output_size=5, W=W)
    >>> output = get_output(l1, x)
    >>> f = theano.function([x], output)
    >>> x_test = np.array([[0, 2], [1, 2]]).astype('int32')
    >>> f(x_test)
    array([[[  0.,   1.,   2.,   3.,   4.],
            [ 10.,  11.,  12.,  13.,  14.]],
    <BLANKLINE>
           [[  5.,   6.,   7.,   8.,   9.],
            [ 10.,  11.,  12.,  13.,  14.]]], dtype=float32)
    """
    def __init__(self, incoming, input_size, output_size,
                 zero_out=0,
                 W=init.Normal(), **kwargs):
        super(EmbeddingLayer, self).__init__(incoming, max_inputs=10, **kwargs)

        self.input_size = input_size
        self.output_size = output_size
        self.zero_out = zero_out
        if output_size == 1:
            self.W = self.add_param(W, (input_size - zero_out, ),
                                    name="W")
        else:
            self.W = self.add_param(W, (input_size - zero_out, output_size),
                                    name="W")

    def get_output_shapes_for(self, input_shapes):
        return tuple(s + (self.output_size, ) for s in input_shapes)

    def get_outputs_for(self, inputs, **kwargs):
        if self.zero_out == 0:
            W = self.W
        elif self.output_size == 1:
            W = T.concatenate((T.zeros((self.zero_out, )), self.W), axis=0)
        else:
            W = T.concatenate((T.zeros((self.zero_out, self.output_size)),
                               self.W), axis=0)
        return tuple(W[i] for i in inputs)
