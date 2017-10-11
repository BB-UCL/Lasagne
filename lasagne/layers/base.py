from collections import OrderedDict
from warnings import warn

import theano.tensor as T

from .helper import get_outputs, get_all_layers
from .. import utils
from ..utils import SCOPE_DELIMITER


__all__ = [
    "Layer",
    "IndexLayer"
]


# Layer base class

class Layer(object):
    """
    The :class:`Layer` class represents a single layer of a neural network. It
    should be subclassed when implementing new types of layers.

    Because each layer can keep track of the layer(s) feeding into it, a
    network's output :class:`Layer` instance can double as a handle to the full
    network.

    Parameters
    ----------
    incoming :
        1. a :class:`Layer` instance
        2. a tuple of :class:`Layer` instances
        3. a list or tuple of the input shape
        4. a list/tuple of lists/tuples with multiple input shapes
    name : a string or None
        An optional name to attach to this layer.
    """
    def __init__(self, incoming, max_inputs=1, num_outputs=1,
                 inner_layers=None, name=None, prefix=None, **kwargs):
        if isinstance(incoming, (list, tuple)):
            if all(isinstance(i, int) or i is None for i in incoming):
                if len(incoming) > 0:
                    # Single shape
                    incoming = (incoming,)
                # Empty means input layer
        else:
            # Single layer
            incoming = (incoming,)

        self.input_shapes = ()
        layers = []
        for l in incoming:
            if isinstance(l, (tuple, list)):
                if not all(isinstance(i, int) or i is None for i in l):
                    raise ValueError("The shape {} is not valid".format(l))
                # Shape
                self.input_shapes += (l,)
                layers.append(None)
            else:
                # Layer
                self.input_shapes += l.output_shapes
                layers.append(l)

        self.input_layers = tuple(layers)
        if len(self.input_shapes) > max_inputs:
            raise ValueError("Can not create layer with more inputs than %d" %
                             max_inputs)

        self.max_inputs = max_inputs
        self.num_outputs = num_outputs
        self.inner_layers = inner_layers
        self._name = name
        self._prefix = prefix
        self.params = OrderedDict()
        self.get_outputs_kwargs = []
        for shape in self.input_shapes:
            if any(d is not None and d <= 0 for d in shape):
                raise ValueError(("Cannot create Layer with a non-positive "
                                  "input_shape dimension. input_shape=%r, "
                                  "self.name=%r") % (shape, self._name))
        self.name = name

    @property
    def input_shape(self):
        if len(self.input_shapes) > 1:
            raise ValueError("The layer has more than 1 input, "
                             "use `self.input_shapes`.")
        return self.input_shapes[0]

    @property
    def output_shapes(self):
        shapes = self.get_output_shapes_for(self.input_shapes)
        for shape in shapes:
            if any(isinstance(s, T.Variable) for s in shape):
                raise ValueError("%s returned a symbolic output shape from its"
                                 " get_output_shape_for() method: %r. This is "
                                 "not allowed; shapes must be tuples of "
                                 "integers for fixed-size dimensions and Nones"
                                 " for variable dimensions." %
                                 (self.__class__.__name__, shape))
        return shapes

    @property
    def output_shape(self):
        if self.num_outputs > 1:
            raise ValueError("The layer has more than 1 output, "
                             "use `self.output_shapes`.")
        return self.output_shapes[0]

    def get_params(self, unwrap_shared=True, **tags):
        """
        Returns a list of Theano shared variables or expressions that
        parameterize the layer.
        !!! Together with the parameters of any innner layers !!!

        By default, all shared variables that participate in the forward pass
        will be returned (in the order they were registered in the Layer's
        constructor via :meth:`add_param()`). The list can optionally be
        filtered by specifying tags as keyword arguments. For example,
        ``trainable=True`` will only return trainable parameters, and
        ``regularizable=True`` will only return parameters that can be
        regularized (e.g., by L2 decay).

        If any of the layer's parameters was set to a Theano expression instead
        of a shared variable, `unwrap_shared` controls whether to return the
        shared variables involved in that expression (``unwrap_shared=True``,
        the default), or the expression itself (``unwrap_shared=False``). In
        either case, tag filtering applies to the expressions, considering all
        variables within an expression to be tagged the same.

        Parameters
        ----------
        unwrap_shared : bool (default: True)
            Affects only parameters that were set to a Theano expression. If
            ``True`` the function returns the shared variables contained in
            the expression, otherwise the Theano expression itself.

        **tags (optional)
            tags can be specified to filter the list. Specifying ``tag1=True``
            will limit the list to parameters that are tagged with ``tag1``.
            Specifying ``tag1=False`` will limit the list to parameters that
            are not tagged with ``tag1``. Commonly used tags are
            ``regularizable`` and ``trainable``.

        Returns
        -------
        list of Theano shared variables or expressions
            A list of variables that parameterize the layer

        Notes
        -----
        For layers without any parameters, this will return an empty list.
        """
        result = list(self.params.keys())

        only = set(tag for tag, value in tags.items() if value)
        if only:
            # retain all parameters that have all of the tags in `only`
            result = [param for param in result
                      if not (only - self.params[param])]

        exclude = set(tag for tag, value in tags.items() if not value)
        if exclude:
            # retain all parameters that have none of the tags in `exclude`
            result = [param for param in result
                      if not (self.params[param] & exclude)]

        if unwrap_shared:
            result = utils.collect_shared_vars(result)

        # Add all of the inner layers parameters
        for l in self.inner_iter:
            result += l.get_params(unwrap_shared=unwrap_shared, **tags)
        return result

    def get_output_shapes_for(self, input_shapes):
        """
        Computes the output shape of this layer, given an input shape.

        Parameters
        ----------
        input_shapes : tuple of tuples
            Each tuple represents the shape of the input. The tuple should have
            as many elements as there are input dimensions, and the elements
            should be integers or `None`.

        Returns
        -------
        tuple of tuples
            Each tuple represents the shape of the output of this layer. The
            tuple has as many elements as there are output dimensions, and the
            elements are all either integers or `None`.

        Notes
        -----
        This method will typically be overridden when implementing a new
        :class:`Layer` class. By default it simply returns the input
        shape. This means that a layer that does not modify the shape
        (e.g. because it applies an elementwise operation) does not need
        to override this method.
        """
        assert len(input_shapes) == len(self.input_shapes)
        return input_shapes

    def get_output_shape_for(self, input_shapes):
        if self.num_outputs > 1:
            raise ValueError("The layer has more than 1 output, "
                             "use `self.output_shapes`.")
        input_shapes = utils.shape_to_tuple(input_shapes)
        return self.get_output_shapes_for(input_shapes)[0]

    def get_outputs_for(self, inputs, **kwargs):
        """
        !!! Should always check if inputs is or not a list
        Propagates the given input through this layer (and only this layer).

        Parameters
        ----------
        inputs : tuple of Theano expression
            The expression to propagate through this layer.

        Returns
        -------
        output : Theano expression
            The output of this layer given the input to this layer.


        Notes
        -----
        This is called by the base :meth:`lasagne.layers.get_output()`
        to propagate data through a network.

        This method should be overridden when implementing a new
        :class:`Layer` class. By default it raises `NotImplementedError`.
        """
        raise NotImplementedError

    def get_output_for(self, inputs, **kwargs):
        if self.num_outputs > 1:
            raise ValueError("The layer has more than 1 output, "
                             "use `self.output_shapes`.")
        inputs = utils.to_tuple(inputs)
        return self.get_outputs_for(inputs, **kwargs)[0]

    def add_param(self, spec, shape, name=None, broadcast_unit_dims=True, **tags):
        """
        Register and possibly initialize a parameter tensor for the layer.

        When defining a layer class, this method is called in the constructor
        to define which parameters the layer has, what their shapes are, how
        they should be initialized and what tags are associated with them.
        This allows layer classes to transparently support parameter
        initialization from numpy arrays and callables, as well as setting
        parameters to existing Theano shared variables or Theano expressions.

        All registered parameters are stored along with their tags in the
        ordered dictionary :attr:`Layer.params`, and can be retrieved with
        :meth:`Layer.get_params()`, optionally filtered by their tags.

        Parameters
        ----------
        spec : Theano shared variable, expression, numpy array or callable
            initial value, expression or initializer for this parameter.
            See :func:`lasagne.utils.create_param` for more information.

        shape : tuple of int
            a tuple of integers representing the desired shape of the
            parameter tensor.

        name : str (optional)
            a descriptive name for the parameter variable. This will be passed
            to ``theano.shared`` when the variable is created, prefixed by the
            layer's name if any (in the form ``'layer_name.param_name'``). If
            ``spec`` is already a shared variable or expression, this parameter
            will be ignored to avoid overwriting an existing name.
        
        broadcast_unit_dims: bool (optional)
             When creating a shared variable whether to make it broadcastable
            along any axis where its size is 1.
        
        **tags (optional)
            tags associated with the parameter can be specified as keyword
            arguments. To associate the tag ``tag1`` with the parameter, pass
            ``tag1=True``.

            By default, the tags ``regularizable`` and ``trainable`` are
            associated with the parameter. Pass ``regularizable=False`` or
            ``trainable=False`` respectively to prevent this.

        Returns
        -------
        Theano shared variable or Theano expression
            the resulting parameter variable or parameter expression

        Notes
        -----
        It is recommended to assign the resulting parameter variable/expression
        to an attribute of the layer for easy access, for example:

        >>> self.w = self.add_param(w, (2, 3), name='W')  #doctest: +SKIP
        """
        # prefix the param name with the layer name if it exists
        if name is None:
            name = str(len(self.params))
        if name is not None:
            if self._name is not None:
                name = "%s%s%s" % (self.name, SCOPE_DELIMITER, name)
        # create shared variable, or pass through given variable/expression
        param = utils.create_param(spec, shape, name, broadcast_unit_dims)
        # parameters should be trainable and regularizable by default
        tags['trainable'] = tags.get('trainable', True)
        tags['regularizable'] = tags.get('regularizable', True)
        self.params[param] = set(tag for tag, value in tags.items() if value)

        return param

    @property
    def name(self):
        name = "" if self._prefix is None else \
            self._prefix + SCOPE_DELIMITER
        name += "" if self._name is None else self._name
        return name

    @name.setter
    def name(self, value):
        self._name = value
        if self._prefix is None:
            prefix = "" if self._name is None else self.name
        elif self._name is None:
            prefix = self._prefix
        else:
            prefix = self._prefix + SCOPE_DELIMITER + self._name
        for l in self.inner_iter:
            l.prefix = prefix
        for p in self.params:
            base = p.name.split(SCOPE_DELIMITER)[-1]
            p.name = prefix + SCOPE_DELIMITER + base

    @property
    def prefix(self):
        return self._prefix

    @prefix.setter
    def prefix(self, value):
        self._prefix = value
        self.name = self._name

    @property
    def inner_iter(self):
        if self.inner_layers is not None:
            if isinstance(self.inner_layers, Layer):
                yield self.inner_layers
            elif isinstance(self.inner_layers, (list, tuple)):
                for l in self.inner_layers:
                    yield l
            elif isinstance(self.inner_layers, dict):
                for l in self.inner_layers.values():
                    yield l
            else:
                raise ValueError("`inner_layers` must be either dict, list, tuple or Layer")

    def curvature_propagation(self, optimizer, inputs, outputs, curvature, make_matrix):
        """
        Stochastic Kronecker-Factored Gauss-Newton
        
        The method makes two computations:
            1. Calculates the Q and G factors for any parameters of the layer
                and adds them to the q_map and g_map
            2. Propagates back the curvature information with respect to 
                the input.
        
        
        Parameters
        ----------
        optimizer: KFOptimizer instance
            
        inputs : Tuple of Theano expression
            The inputs of the layer.

        outputs : Tuple of Theano expression
            The outputs of the layer.

        curvature : Tuple of Theano expression
            The activation square-roots of the activation Gauss-Newton matrices
            with respect to the outputs of the layer. 
        
        make_matrix: callable
            A function which to construct and store the Kronecker-Factored
            matrix kron(A,B) for the parameters in this layer.
            the following signature:
            make_matrix(layer, a_dim, b_dim, a, b, params, name)
                layer - the owning layer of the parameters
                a_dim - dimensionality of A
                b_dim - dimensionality of A
                a/a_sqrt - Factor or square root of the factor
                b/b_sqrt - Factor or square root of the factor
                params - the parameters corresponding to.
                name - any extra name to add except the layer's.

        Returns
        -------
        Theano expressions for the Gauss-Newton matrices with respect to the 
        inputs of the layer.
        
        By default the method is implemented for non-parametric layers and just
        propagates the Jacobian-vector product
        """
        assert len(inputs) == len(self.input_shapes)
        assert len(outputs) == self.num_outputs
        assert len(curvature) == self.num_outputs
        warn("Using default implementation of SKFGN for {}."
             .format(str(self.__class__)))
        if optimizer.variant == "kfra":
            raise NotImplementedError
        else:
            return T.Lop(outputs, inputs, curvature, disconnected_inputs="warn")

    def params_to_fused(self, params):
        """
        Converts a list of expression of the same shapes as the parameters of
        the layer to a fused version.
        This is used for SKFGN for when passing gradients, but having a single

        Parameters
        ----------
        params: list or tuple
            List of expressions to fuse.

        Returns
        -------
        An expression of all of the inputs fused together.
        """
        return None

    def params_from_fused(self, fused, num):
        """
        Reverses the effect of :params_to_fused:

        Parameters
        ----------
        fused: Theano expression
            Expression of fused variables.

        Returns
        -------
        The un-fused expression.
        """
        return None

    # def serialize(self):
    #     raise NotImplementedError

    def base_serialize(self):
        kwargs = OrderedDict()
        kwargs["type"] = self.__class__.__name__
        kwargs["name"] = self._name
        kwargs["prefix"] = self._prefix
        kwargs["incomming"] = [l if l is not None else self.input_shapes[i] for i, l in enumerate(self.input_layers)]
        return kwargs

    def base_pretty_print(self, indent=""):
        incoming = tuple(l.name if l is not None else None for l in self.input_layers)
        incoming = incoming[0] if len(incoming) == 1 else incoming
        shape = self.get_output_shapes_for(self.input_shapes)
        if len(shape) == 1:
            print("{}{:<10} {} -> {}".format(indent, self.name, incoming, shape[0]))
            for l in self.inner_iter:
                l.base_pretty_print(indent + "\t")
        elif len(shape) == 2:
            print("{}{:<10} {} -> {}, {}".format(indent, self.name, incoming, shape[0], shape[1]))
            for l in self.inner_iter:
                l.base_pretty_print(indent + "\t")

    def __call__(self, *args, **kwargs):
        assert len(args) < 2
        if len(args) == 0:
            layers = {l.name: l for l in get_all_layers(self)}
            inputs = {layers[k]: v for k, v in kwargs.items() if layers.get(k) is not None}
            kwargs = dict(filter(lambda pair: layers.get(pair[0]) is None, kwargs.items()))
            for k in inputs.keys():
                if layers.get(k) is not None:
                    kwargs.pop(k)
            outs = get_outputs(self, inputs=inputs, **kwargs)
        else:
            outs = get_outputs(self, inputs=args[0], **kwargs)
        if len(outs) == 1:
            return outs[0]
        else:
            return outs


class IndexLayer(Layer):
    def __init__(self, incoming, indexes, including=True, **kwargs):
        super(IndexLayer, self).__init__(incoming, max_inputs=100, **kwargs)
        if not isinstance(indexes, (list, tuple)):
            indexes = (indexes, )
        for i in indexes:
            if i >= len(self.input_shapes):
                raise ValueError("Index provided - {}, but have only {} inputs."
                                 .format(i, len(self.input_shapes)))
        if including:
            self.indexes = indexes
        else:
            self.indexes = tuple(i for i in range(len(self.input_shapes))
                                 if i not in indexes)

    def get_output_shapes_for(self, input_shapes):
        assert len(input_shapes) == len(self.input_shapes)
        return tuple(input_shapes[i] for i in self.indexes)

    def get_outputs_for(self, inputs, **kwargs):
        assert len(inputs) == len(self.input_shapes)
        return tuple(inputs[i] for i in self.indexes)

    # def serialize(self):
    #     kwargs = self.base_serialize()
    #     kwargs["indexes"] = self.indexes
    #     return kwargs

    # def serialize(self):
    #
    #     raise NotImplementedError
    #
    # @staticmethod
    # def deserialize(layer):
    #     raise NotImplementedError
