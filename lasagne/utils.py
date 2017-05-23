import numpy as np

import theano
from theano import tensor as T
from theano.printing import Print


SCOPE_DELIMITER = "::"


def floatX(arr):
    """Converts data to a numpy array of dtype ``theano.config.floatX``.

    Parameters
    ----------
    arr : array_like
        The data to be converted.

    Returns
    -------
    numpy ndarray
        The input array in the ``floatX`` dtype configured for Theano.
        If `arr` is an ndarray of correct dtype, it is returned as is.
    """
    return np.asarray(arr, dtype=theano.config.floatX)


def th_fx(var):
    """Converts the input variable to a symbolic Theano expression 
    of dtype ``theano.config.floatX``.

    Parameters
    ----------
        var : Theano expression, Theano variable or numpy array
            The expression to be converted.

    Returns
    -------
        A symbolic Theano expression representing the cast of the
        variable to ``floatX``
    """
    return T.cast(var, theano.config.floatX)


def shared_empty(dim=2, dtype=None):
    """Creates empty Theano shared variable.

    Shortcut to create an empty Theano shared variable with
    the specified number of dimensions.

    Parameters
    ----------
    dim : int, optional
        The number of dimensions for the empty variable, defaults to 2.
    dtype : a numpy data-type, optional
        The desired dtype for the variable. Defaults to the Theano
        ``floatX`` dtype.

    Returns
    -------
    Theano shared variable
        An empty Theano shared variable of dtype ``dtype`` with
        `dim` dimensions.
    """
    if dtype is None:
        dtype = theano.config.floatX

    shp = tuple([1] * dim)
    return theano.shared(np.zeros(shp, dtype=dtype))


def as_theano_expression(input):
    """Wrap as Theano expression.

    Wraps the given input as a Theano constant if it is not
    a valid Theano expression already. Useful to transparently
    handle numpy arrays and Python scalars, for example.

    Parameters
    ----------
    input : number, numpy array or Theano expression
        Expression to be converted to a Theano constant.

    Returns
    -------
    Theano symbolic constant
        Theano constant version of `input`.
    """
    if isinstance(input, theano.gof.Variable):
        return input
    else:
        try:
            return theano.tensor.constant(input)
        except Exception as e:
            raise TypeError("Input of type %s is not a Theano expression and "
                            "cannot be wrapped as a Theano constant (original "
                            "exception: %s)" % (type(input), e))


def collect_shared_vars(expressions):
    """Returns all shared variables the given expression(s) depend on.

    Parameters
    ----------
    expressions : Theano expression or iterable of Theano expressions
        The expressions to collect shared variables from.

    Returns
    -------
    list of Theano shared variables
        All shared variables the given expression(s) depend on, in fixed order
        (as found by a left-recursive depth-first search). If some expressions
        are shared variables themselves, they are included in the result.
    """
    # wrap single expression in list
    if isinstance(expressions, theano.Variable):
        expressions = [expressions]
    # return list of all shared variables
    return [v for v in theano.gof.graph.inputs(reversed(expressions))
            if isinstance(v, theano.compile.SharedVariable)]


def one_hot(x, m=None):
    """One-hot representation of integer vector.

    Given a vector of integers from 0 to m-1, returns a matrix
    with a one-hot representation, where each row corresponds
    to an element of x.

    Parameters
    ----------
    x : integer vector
        The integer vector to convert to a one-hot representation.
    m : int, optional
        The number of different columns for the one-hot representation. This
        needs to be strictly greater than the maximum value of `x`.
        Defaults to ``max(x) + 1``.

    Returns
    -------
    Theano tensor variable
        A Theano tensor variable of shape (``n``, `m`), where ``n`` is the
        length of `x`, with the one-hot representation of `x`.

    Notes
    -----
    If your integer vector represents target class memberships, and you wish to
    compute the cross-entropy between predictions and the target class
    memberships, then there is no need to use this function, since the function
    :func:`lasagne.objectives.categorical_crossentropy()` can compute the
    cross-entropy from the integer vector directly.

    """
    if m is None:
        m = T.cast(T.max(x) + 1, 'int32')

    return T.eye(m)[T.cast(x, 'int32')]


def unique(l):
    """Filters duplicates of iterable.

    Create a new list from l with duplicate entries removed,
    while preserving the original order.

    Parameters
    ----------
    l : iterable
        Input iterable to filter of duplicates.

    Returns
    -------
    list
        A list of elements of `l` without duplicates and in the same order.
    """
    new_list = []
    seen = set()
    for el in l:
        if el not in seen:
            new_list.append(el)
            seen.add(el)

    return new_list


def as_tuple(x, N, t=None):
    """
    Coerce a value to a tuple of given length (and possibly given type).

    Parameters
    ----------
    x : value or iterable
    N : integer
        length of the desired tuple
    t : type, optional
        required type for all elements

    Returns
    -------
    tuple
        ``tuple(x)`` if `x` is iterable, ``(x,) * N`` otherwise.

    Raises
    ------
    TypeError
        if `type` is given and `x` or any of its elements do not match it
    ValueError
        if `x` is iterable, but does not have exactly `N` elements
    """
    try:
        X = tuple(x)
    except TypeError:
        X = (x,) * N

    if (t is not None) and not all(isinstance(v, t) for v in X):
        raise TypeError("expected a single value or an iterable "
                        "of {0}, got {1} instead".format(t.__name__, x))

    if len(X) != N:
        raise ValueError("expected a single value or an iterable "
                         "with length {0}, got {1} instead".format(N, x))

    return X


def inspect_kwargs(func):
    """
    Inspects a callable and returns a list of all optional keyword arguments.

    Parameters
    ----------
    func : callable
        The callable to inspect

    Returns
    -------
    kwargs : list of str
        Names of all arguments of `func` that have a default value, in order
    """
    # We try the Python 3.x way first, then fall back to the Python 2.x way
    try:
        from inspect import signature
    except ImportError:  # pragma: no cover
        from inspect import getargspec
        spec = getargspec(func)
        return spec.args[-len(spec.defaults):] if spec.defaults else []
    else:  # pragma: no cover
        params = signature(func).parameters
        return [p.name for p in params.values() if p.default is not p.empty]


def compute_norms(array, norm_axes=None):
    """ Compute incoming weight vector norms.

    Parameters
    ----------
    array : numpy array or Theano expression
        Weight or bias.
    norm_axes : sequence (list or tuple)
        The axes over which to compute the norm.  This overrides the
        default norm axes defined for the number of dimensions
        in `array`. When this is not specified and `array` is a 2D array,
        this is set to `(0,)`. If `array` is a 3D, 4D or 5D array, it is
        set to a tuple listing all axes but axis 0. The former default is
        useful for working with dense layers, the latter is useful for 1D,
        2D and 3D convolutional layers.
        Finally, in case `array` is a vector, `norm_axes` is set to an empty
        tuple, and this function will simply return the absolute value for
        each element. This is useful when the function is applied to all
        parameters of the network, including the bias, without distinction.
        (Optional)

    Returns
    -------
    norms : 1D array or Theano vector (1D)
        1D array or Theano vector of incoming weight/bias vector norms.

    Examples
    --------
    >>> array = np.random.randn(100, 200)
    >>> norms = compute_norms(array)
    >>> norms.shape
    (200,)

    >>> norms = compute_norms(array, norm_axes=(1,))
    >>> norms.shape
    (100,)
    """

    # Check if supported type
    if not isinstance(array, theano.Variable) and \
       not isinstance(array, np.ndarray):
        raise RuntimeError(
            "Unsupported type {}. "
            "Only theano variables and numpy arrays "
            "are supported".format(type(array))
        )

    # Compute default axes to sum over
    ndim = array.ndim
    if norm_axes is not None:
        sum_over = tuple(norm_axes)
    elif ndim == 1:          # For Biases that are in 1d (e.g. b of DenseLayer)
        sum_over = ()
    elif ndim == 2:          # DenseLayer
        sum_over = (0,)
    elif ndim in [3, 4, 5]:  # Conv{1,2,3}DLayer
        sum_over = tuple(range(1, ndim))
    else:
        raise ValueError(
            "Unsupported tensor dimensionality {}. "
            "Must specify `norm_axes`".format(array.ndim)
        )

    # Run numpy or Theano norm computation
    if isinstance(array, theano.Variable):
        # Apply theano version if it is a theano variable
        if len(sum_over) == 0:
            norms = T.abs_(array)   # abs if we have nothing to sum over
        else:
            norms = T.sqrt(T.sum(array**2, axis=sum_over))
    elif isinstance(array, np.ndarray):
        # Apply the numpy version if ndarray
        if len(sum_over) == 0:
            norms = abs(array)     # abs if we have nothing to sum over
        else:
            norms = np.sqrt(np.sum(array**2, axis=sum_over))

    return norms


def create_param(spec, shape, name=None, broadcast_unit_dims=True):
    """
    Helper method to create Theano shared variables for layer parameters
    and to initialize them.

    Parameters
    ----------
    spec : scalar number, numpy array, Theano expression, or callable
        Either of the following:

        * a scalar or a numpy array with the initial parameter values
        * a Theano expression or shared variable representing the parameters
        * a function or callable that takes the desired shape of
          the parameter array as its single argument and returns
          a numpy array, a Theano expression, or a shared variable
          representing the parameters.

    shape : iterable of int
        a tuple or other iterable of integers representing the desired
        shape of the parameter array.

    name : string, optional
        The name to give to the parameter variable. Ignored if `spec`
        is or returns a Theano expression or shared variable that
        already has a name.
    
    broadcast_unit_dims: bool, optional
        When creating a shared variable whether to make it broadcastable
        along any axis where its size is 1.

    Returns
    -------
    Theano shared variable or Theano expression
        A Theano shared variable or expression representing layer parameters.
        If a scalar or a numpy array was provided, a shared variable is
        initialized to contain this array. If a shared variable or expression
        was provided, it is simply returned. If a callable was provided, it is
        called, and its output is used to initialize a shared variable.

    Notes
    -----
    This function is called by :meth:`Layer.add_param()` in the constructor
    of most :class:`Layer` subclasses. This enables those layers to
    support initialization with scalars, numpy arrays, existing Theano shared
    variables or expressions, and callables for generating initial parameter
    values, Theano expressions, or shared variables.
    """
    import numbers  # to check if argument is a number
    shape = tuple(shape)  # convert to tuple if needed
    if any(d <= 0 for d in shape):
        raise ValueError((
            "Cannot create param with a non-positive shape dimension. "
            "Tried to create param with shape=%r, name=%r") % (shape, name))

    err_prefix = "cannot initialize parameter %s: " % name
    if callable(spec):
        spec = spec(shape)
        err_prefix += "the %s returned by the provided callable"
    else:
        err_prefix += "the provided %s"

    if isinstance(spec, numbers.Number) or isinstance(spec, np.generic) \
            and spec.dtype.kind in 'biufc':
        spec = np.asarray(spec)

    if isinstance(spec, np.ndarray):
        if spec.shape != shape:
            raise ValueError("%s has shape %s, should be %s" %
                             (err_prefix % "numpy array", spec.shape, shape))
        # We assume parameter variables do not change shape after creation.
        # We can thus fix their broadcast pattern, to allow Theano to infer
        # broadcastable dimensions of expressions involving these parameters.
        if broadcast_unit_dims:
            bcast = tuple(s == 1 for s in shape)
            spec = theano.shared(spec, broadcastable=bcast)
        else:
            spec = theano.shared(spec)

    if isinstance(spec, theano.Variable):
        # We cannot check the shape here, Theano expressions (even shared
        # variables) do not have a fixed compile-time shape. We can check the
        # dimensionality though.
        if spec.ndim != len(shape):
            raise ValueError("%s has %d dimensions, should be %d" %
                             (err_prefix % "Theano variable", spec.ndim,
                              len(shape)))
        # We only assign a name if the user hasn't done so already.
        if not spec.name:
            spec.name = name
        return spec

    else:
        if "callable" in err_prefix:
            raise TypeError("%s is not a numpy array or a Theano expression" %
                            (err_prefix % "value"))
        else:
            raise TypeError("%s is not a numpy array, a Theano expression, "
                            "or a callable" % (err_prefix % "spec"))


def unroll_scan(fn, sequences, outputs_info, non_sequences, n_steps,
                go_backwards=False):
        """
        Helper function to unroll for loops. Can be used to unroll theano.scan.
        The parameter names are identical to theano.scan, please refer to here
        for more information.

        Note that this function does not support the truncate_gradient
        setting from theano.scan.

        Parameters
        ----------

        fn : function
            Function that defines calculations at each step.

        sequences : TensorVariable or list of TensorVariables
            List of TensorVariable with sequence data. The function iterates
            over the first dimension of each TensorVariable.

        outputs_info : list of TensorVariables
            List of tensors specifying the initial values for each recurrent
            value.

        non_sequences: list of TensorVariables
            List of theano.shared variables that are used in the step function.

        n_steps: int
            Number of steps to unroll.

        go_backwards: bool
            If true the recursion starts at sequences[-1] and iterates
            backwards.

        Returns
        -------
        List of TensorVariables. Each element in the list gives the recurrent
        values at each time step.

        """
        if not isinstance(sequences, (list, tuple)):
            sequences = [sequences]

        # When backwards reverse the recursion direction
        counter = range(n_steps)
        if go_backwards:
            counter = counter[::-1]

        output = []
        prev_vals = outputs_info
        for i in counter:
            step_input = [s[i] for s in sequences] + prev_vals + non_sequences
            out_ = fn(*step_input)
            # The returned values from step can be either a TensorVariable,
            # a list, or a tuple.  Below, we force it to always be a list.
            if isinstance(out_, T.TensorVariable):
                out_ = [out_]
            if isinstance(out_, tuple):
                out_ = list(out_)
            output.append(out_)

            prev_vals = output[-1]

        # iterate over each scan output and convert it to same format as scan:
        # [[output11, output12,...output1n],
        # [output21, output22,...output2n],...]
        output_scan = []
        for i in range(len(output[0])):
            l = map(lambda x: x[i], output)
            output_scan.append(T.stack(*l))

        return output_scan


def to_tuple(x):
    """
    Helper function that converts x to a tuple (x, ) if x is not a list or
    tuple otherwise returns x.
    
    Parameters
    ----------

    x : Any python expression
        Expression to convert
    
    Returns
    -------
    Either x or (x ,)
    """
    if isinstance(x, (tuple, list)):
        return x
    else:
        return x,


def shape_to_tuple(shape_or_shapes):
    """
    Helper function that converts the input to a tuple of shapes
    if it is not already that.

    Parameters
    ----------

    shape_or_shapes : A list, tuple or Theano expression
        Expression to convert

    Returns
    -------
    Either shape_or_shapes or (shape_or_shapes, )
    """
    if isinstance(shape_or_shapes, (T.TensorVariable, T.Variable)):
        return shape_or_shapes,
    elif not isinstance(shape_or_shapes, (tuple, list)):
        raise ValueError("Input must tuple, list or Theano expression.")
    elif isinstance(shape_or_shapes[0], int) or shape_or_shapes[0] is None:
        return shape_or_shapes,
    else:
        return shape_or_shapes


def extract_clean_dims(shape):
    return tuple(s for s in shape if s is not None)


def broadcast_to_none(shape):
    """
    Helper function which returns the pattern needed to be passed to
    dimshuffle for a variable having only the non None dimensions of
    the shape

    Parameters
    ----------

    shape : A list or tuple
        The shape for which the pattern is needed

    Returns
    -------
    The pattern for dimshuffle
    """
    pattern = ()
    c = 0
    for s in shape:
        if s is None:
            pattern += ("x",)
        else:
            pattern += (c,)
            c += 1
    return pattern


def theano_print_shape(var, msg):
    """
    Helper function for printing the shape of a Theano expression during run
    time of the Theano graph.

    Parameters
    ----------

    var : Theano expression
        The variable whose shape to be printed at runtime.
    
    msg : str 
        The message to be printed together with the shape.

    Returns
    -------
    A Theano expression which should be used instead of the original expression
    in order the printing to happen.
    """
    pr = Print(msg)(T.shape(var))
    return T.switch(T.lt(0, 1), var, T.cast(pr[0], var.dtype))


def theano_print_min_max(var, msg):
    """
    Helper function for printing the min and the max values of a 
    Theano expression during run time of the Theano graph.

    Parameters
    ----------

    var : Theano expression
        The variable whose mi and max values to be printed at runtime.

    msg : str 
        The message to be printed together min and max.

    Returns
    -------
    A Theano expression which should be used instead of the original expression
    in order the printing to happen.
    """
    pr = Print(msg)(T.stack((T.min(var), T.max(var))))
    return T.switch(T.lt(0, 1), var, T.cast(pr[0], var.dtype))


def theano_print_values(var, msg):
    """
    Helper function for printing the all of the values of a 
    Theano expression during run time of the Theano graph.

    Parameters
    ----------

    var : Theano expression
        The variable whose mi and max values to be printed at runtime.

    msg : str 
        The message to be printed together values.

    Returns
    -------
    A Theano expression which should be used instead of the original expression
    in order the printing to happen.
    """
    return Print(msg)(var)


def frobenius_norm(var):
    """
    Calculates the Frobenius norm of the input using Theano.
    
    The input must be a two dimensional matrix.

    Parameters
    ----------

    var : numpy array or Theano expression
        The matrix for which to calculate the norm.

    Returns
    -------
    Theano expression of the Frobenius norm of the input.
    """
    assert var.ndim == 2
    return T.sqrt(T.sum(T.sqr(var)))


def mean_frobenius_norm(var):
    """
    Calculates the Mean Frobenius norm of the input using Theano.
    This is the Frobenius norm divided inside the square root by number of 
    columns of the matrix.

    The input must be a two dimensional matrix.

    Parameters
    ----------

    var : numpy array or Theano expression
        The matrix for which to calculate the norm.

    Returns
    -------
    Theano expression of the Mean Frobenius of the input.
    """
    assert var.ndim == 2
    return T.sqrt(T.mean(T.sum(T.sqr(var), axis=1)))


def trace_norm(var):
    """
    Calculates the Trace norm of the input using Theano.

    The input must be a two dimensional matrix.

    Parameters
    ----------

    var : numpy array or Theano expression
        The matrix for which to calculate the norm.

    Returns
    -------
    Theano expression of the Trace norm of the input.
    """
    assert var.ndim == 2
    return T.sum(T.diag(var))


def mean_trace_norm(var):
    """
    Calculates the Mean Trace norm of the input using Theano.
    This is the Trace norm divided by number of columns of the matrix.
    
    The input must be a two dimensional matrix.

    Parameters
    ----------

    var : numpy array or Theano expression
        The matrix for which to calculate the norm.

    Returns
    -------
    Theano expression of the Mean Trace of the input.
    """
    assert var.ndim == 2
    return T.mean(T.diag(var))


def gauss_newton_product(outputs, outputs_hess, params, v1, v2=None):
    """
    Calculates the quadratic product between the Generalized Gauss-Newton 
    matrix and two vectors:
    
    :math:`v_1^T G v_2`
    
    If v2 is None than v2=v1.
    
    The first dimension is assumed to be mini-batch dimension.

    Parameters
    ----------
    
    outputs: Theano expression
        The outputs of the function for which to compute the product.
        
    outputs_hess : Theano expression
        The Hessian matrix of the objective with respect to each data point.
        If this is two dimensional, than it is assumed as the diagonal of 
        the Hessian.
        !!! Currently only two dimensional is supported !!!.

    Returns
    -------
    The expected value of the product with the Generalized Gauss-Newton matrix.
    """
    assert outputs_hess.ndim == 2
    Jv1 = T.Rop(outputs, params, v1)
    Jv2 = T.Rop(outputs, params, v2) if v2 else Jv1
    return T.mean(T.sum(Jv1 * Jv2 * outputs_hess, axis=1))


def linear_solve(A, b, A_structure="general"):
    """
    Solves the linear system:

    :math:`Ax = b`

    using the correct Theano Op based on the current device.
        
    This is needed as Theano's optimization currently does not pass the 
    A_structure when optimizing the standard Op to the GpuOp.

    Parameters
    ----------

    A: Theano expression
        Must be a square matrix.
    
    b: Theano expression
        Must be a matrix.
        
    A_structure : str
        One of 'general' or 'symmetric'.

    Returns
    -------
    The solution to the linear system computed on the current Theano device.
    """
    if theano.config.device.startswith("cpu"):
        from theano.tensor.slinalg import Solve
        return Solve(A_structure=A_structure)(A, b)
    else:
        from theano.gpuarray.linalg import GpuCusolverSolve
        return GpuCusolverSolve(A_structure=A_structure)(A, b)


def expand_variable(var, repeats, axis=0):
    """
    Helper function to expand a variable number of times by repeating
    it along the given axis, such that the result along on that axis is:
    [var1, var1, var1, ..., var2, var2, var2, ..., varN, varN, varN, ...]

    Parameters
    ----------

    var : TensorVariable
        The variable which to be repeated/expanded
    
    repeats: int
        Number of repeats.
        
    axis: int
        Axis along which to expand the variable.
    
    Returns
    -------
    The expanded variable.
    """
    if repeats == 1:
        return var
    shape = [var.shape[i] for i in range(var.ndim)]
    extra_shape = shape[:axis] + [1] + shape[axis:]
    repeats = [1 for _ in shape[:axis]] + [repeats] + [1 for _ in shape[axis:]]
    expanded = T.tile(var.reshape(extra_shape), repeats)
    shape[axis] *= repeats[axis]
    return expanded.reshape(shape)


def collapse_variable(var, repeats, axis=0):
    """
    Reverses the expand_variable function by taking the mean of the repeats.

    Parameters
    ----------

    var : TensorVariable
        The variable which to be collapsed

    repeats: int
        Number of repeats.

    axis: int
        Axis along which to collapse the variable.

    Returns
    -------
    The collapsed variable.
    """
    if repeats == 1:
        return var
    shape = [var.shape[i] for i in range(var.ndim)]
    extra_shape = shape[:axis] + [repeats] + shape[axis:]
    extra_shape[axis] = T.int_div(extra_shape[axis], repeats)
    var = T.reshape(var, extra_shape)
    return T.mean(var, axis=(axis + 1))


def dfs_path(f, w):
    """
    Performs a Depth-First-Search trying to find the dependence path
    of Theano expression of f with respect to w.

    Parameters
    ----------

    f : TensorVariable
        The source variable from which the DFS search starts.

    w : TensorVariable
        The variable to which we need to find the path. When this variable
        is found the search stops and the path is returned.

    Returns
    -------
    The path from f->w including the two expression. This means that 
    f == path[0] and w == path[-1]
    """
    queue = [[f]]
    while len(queue) > 0:
        path = queue.pop(0)
        current = path[-1]
        if current == w:
            return path
        if current.owner is not None:
            for child in current.owner.inputs:
                new_path = list(path) + [child]
                queue.append(new_path)
    raise ValueError("Did not find the variable {} in the graph.".format(str(w)))
