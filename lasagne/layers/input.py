from collections import OrderedDict

import theano
import theano.tensor as T

from .base import Layer


__all__ = [
    "InputLayer",
    "ParameterLayer"
]


class InputLayer(Layer):
    """
    This layer holds a symbolic variable that represents a network input. A
    variable can be specified when the layer is instantiated, else it is
    created.

    Parameters
    ----------
    shape : tuple of `int` or `None` elements
        The shape of the input. Any element can be `None` to indicate that the
        size of that dimension is not fixed at compile time.

    input_var : Theano symbolic variable or `None` (default: `None`)
        A variable representing a network input. If it is not provided, a
        variable will be created.

    Raises
    ------
    ValueError
        If the dimension of `input_var` is not equal to `len(shape)`

    Notes
    -----
    The first dimension usually indicates the batch size. If you specify it,
    Theano may apply more optimizations while compiling the training or
    prediction function, but the compiled function will not accept data of a
    different batch size at runtime. To compile for a variable batch size, set
    the first shape element to `None` instead.

    Examples
    --------
    >>> from lasagne.layers import InputLayer
    >>> l_in = InputLayer((100, 20))
    """
    def __init__(self, shape, input_var=None, name=None, dtype=None, **kwargs):
        self.shape = tuple(shape)
        if any(d is not None and d <= 0 for d in self.shape):
            raise ValueError(("Cannot create InputLayer with a non-positive "
                              "shape dimension. shape=%r, self.name=%r") %
                             (self.shape, name))

        ndim = len(self.shape)
        if input_var is None:
            # create the right TensorType for the given dimensionality/shape
            dtype = theano.config.floatX if dtype is None else dtype
            input_var_type = T.TensorType(dtype, [s == 1 for s in self.shape])
            var_name = ("%s.input" % name) if name is not None else "input"
            input_var = input_var_type(var_name)
        else:
            # ensure the given variable has the correct dimensionality
            if input_var.ndim != ndim:
                raise ValueError("shape has %d dimensions, but variable has "
                                 "%d" % (ndim, input_var.ndim))
        self.input_var = input_var
        super(InputLayer, self).__init__((), max_inputs=0, name=name, **kwargs)

    @Layer.output_shapes.getter
    def output_shapes(self):
        return self.shape,

    def base_pretty_print(self, indent=""):
        print("{}{:<10} {}".format(indent, self.name, self.shape))


class ParameterLayer(Layer):
    def __init__(self, init_value, dtype=None, **kwargs):
        super(ParameterLayer, self).__init__((), max_inputs=0, **kwargs)
        dtype = theano.config.floatX if dtype is None else dtype
        self.p = self.add_param(init_value.astype(dtype), init_value.shape, name="param")


    @Layer.output_shapes.getter
    def output_shapes(self):
        return self.p.get_value().shape,

    def get_outputs_for(self, inputs, **kwargs):
        assert len(inputs) == 0
        return self.p,

    def base_pretty_print(self, indent=""):
        print("{}{:<10} {}".format(indent, self.name, self.shape))