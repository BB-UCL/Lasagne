import abc
import itertools

import numpy as np
import scipy.linalg as la

import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import lasagne as nn
from lasagne import layers as L


class CurvatureApproximator(metaclass=abc.ABCMeta):
    """Base class for numerically calculating an approximation to the curvature of a network."""

    def __init__(self, invar, loss_layer):
        """
        invar: the symbolic input to your computational graph, e.g. the matrix for your minibatch. We generally use
               model Fisher, hence the labels are not needed
        loss_layer: the loss layer of your network (from `lasagne.layers.objectives`)
        """
        self.invar = invar
        self.loss_layer = loss_layer

    def estimate_curvature(self, dataset_or_datasets):
        """
        Calculates the total curvature approximation, i.e. the sum over the data points, not the mean.

        Parameters
        ----------
        dataset_or_datasets: either a dataset or list of datasets. Each dataset needs to define an `iter` method
                             that takes a string for the partition as an argument and a batch size and returns an
                             iterator over minibatches for the dataset.

        Returns
        -------
        List of curvature approximations (numpy arrays), where the length of the list depends on the number of layers
        with parameters that feed into the loss layer. Note that each element can be a tuple, e.g. for Kronecker
        factored approximation.
        """
        if isinstance(dataset_or_datasets, (tuple, list)):
            datasets = dataset_or_datasets
        else:
            datasets = (dataset_or_datasets,)

        return self.estimate_total_curvature(datasets)

    @abc.abstractmethod
    def estimate_total_curvature(self, datasets):
        return


def calculate_curvature_cholesky(qs, gs, reg=0, scale=1, norm=np.trace):
    """
    Helper function for calculating the Choleksy decomposition of the KF curvature factors. As

    Parameters
    ----------
    qs: list of input covariances to each layer. We calculate the upper cholesky of those, as we will solve the linear
        system for these factors when sampling from a matrix normal (we calculate the precision rather than the
        covariance for the distribution, i.e. the inverse, which reverses the order of the square roots.
    gs: list of activation Hessian/Fishers/Gauss-Newton for each layer.
    reg: Multiplier for the identity that is added to each factor for regularization.
    scale: Multipler for the total scale of the matrix for regularization. Each curvature factor is multiplied by the
           square root of this number.
    norm: Norm for normalizing the factors w.r.t. each other (see section 6.3 in the KFAC paper). If `norm=None`, the
          qs and gs remain unchanged, for `norm=np.trace`, their traces are set to be equal.
    """
    assert len(qs) == len(gs)
    if norm is not None:
        omegas = [np.sqrt((norm(q) * g.shape[0]) / norm(g) * q.shape[0])
                  for q, g in zip(qs, gs)]
    else:
        omegas = [1] * len(qs)

    qc = [theano.shared(la.cholesky(np.sqrt(scale) * q / o + reg * np.eye(q.shape[0]),
                                    lower=False).astype(theano.config.floatX))
          for q, o in zip(qs, omegas)]
    gc = [theano.shared(la.cholesky(np.sqrt(scale) * g * o + reg * np.eye(g.shape[0]),
                                    lower=True).astype(theano.config.floatX))
          for g, o in zip(gs, omegas)]

    return qc, gc


class KFCurvatureCalculator(CurvatureApproximator):
    """
    Class for numerically calculating the block-diagonal Kronecker factored curvautre approximation of a network.
    """

    def __init__(self, invar, loss_layer, backprop_variant="skfgn-fisher", batch_size=250, **output_kwargs):
        """
        invar: as in the base class
        loss_layer: as in the baseclass
        backprop_variant: see `VARIANTS` in `skfgn.py`
        batch_size: batch size for iterating over the datasets
        output_kwargs: any arguments that need to be passed to `L.get_output`, e.g. `deterministic=True`
        """
        super(KFCurvatureCalculator, self).__init__(invar, loss_layer)
        self.variant = backprop_variant
        self.batch_size = batch_size
        self.output_kwargs = output_kwargs

        self.calculate_curvature = self._compile_curvature_calculation()

    def estimate_total_curvature(self, datasets):
        """
        Calculates the Kronecker factors for each parameter layer feeding into the loss layer. Each factors is scaled
        by the square root of the dataset size, such that the Kronecker product gives the (approximated) average
        curvature of the block times the number of data points.

        Parameters
        ----------
        datasets: as in superclass
        """
        curvature_factors = []
        n = 0
        for dataset in datasets:
            for x, _ in dataset.iter("train", self.batch_size):
                bs = x.shape[0]
                n += bs
                factors = self.calculate_curvature(x.astype(theano.config.floatX))
                if curvature_factors:
                    curvature_factors = [f * bs + cf for f, cf in zip(factors, curvature_factors)]
                else:
                    curvature_factors = [f * bs for f in factors]

        curvature_factors = [f / np.sqrt(n) for f in curvature_factors]
        return curvature_factors

    def _compile_curvature_calculation(self):
        output = self.diagonal_fisher_block_factors()
        flat_output = list(itertools.chain.from_iterable(output))

        return theano.function([self.invar], flat_output)

    def diagonal_fisher_block_factors(self):
        """
        Returns a list of symbolic expressions for the Kronecker factors of the
        diagonol blocks of the Fisher matrix for a list of dense layers. Assumes
        that `skfgn` has been called on each layer.
        """
        kf_matrices = self._kf_matrices()
        return [(m.factors[1].get_raw(), m.factors[0].get_raw()) for m in kf_matrices]

    def _kf_matrices(self):
        """
        Sets up the list of symbolic KF matrices (from output to input).
        """
        optimizer = nn.skfgn.Optimizer(self.variant)
        _, inputs_map, outputs_map = L.get_output(self.loss_layer, get_maps=True, **self.output_kwargs)
        # the curvature propagation returns the matrices ordered from output to
        # input, but since we care about defining a forward pass, it is more
        # convenient to have it the other way around
        return list(reversed(optimizer.curvature_propagation(
            self.loss_layer, inputs_map=inputs_map, outputs_map=outputs_map)))


class DiagonalFisherCalculator(CurvatureApproximator):
    """
    Class for approximating the diagonal entries of the Fisher by summing up the squared gradients per datapoint.
    """

    def __init__(self, invar, loss_layer, rng=None, n_samples=0, **output_kwargs):
        """
        Parameters
        ----------
        invar: see superclass
        loss_layer: see superclass
        rng: random number generator for sampling labels from the network output (for calculating the model fisher)
        n_samples: size of the subsample to draw from the dataset (we're iterating over the dataset one datapoint at
                   a time to sum the squared gradients, which is quite slow, hence it can make sense to only sample
                   a smaller number of datapoints to speed things up. The scale will always be the size of the dataset.)
        output_kwargs: arguments to pass to `L.get_output`, e.g. `deterministic=True`.
        """
        if not isinstance(loss_layer, (L.BinaryLogitsCrossEntropy, L.CategoricalLogitsCrossEntropy)):
            raise ValueError("Diagonal Fisher Calculator is only implemented for Binary/-CategoricalCrossEntropy"
                             " for now, but received '{}' loss layer".format(loss_layer.__class__.__name__))
        super(DiagonalFisherCalculator, self).__init__(invar, loss_layer)
        self.rng = rng if rng is not None else RandomStreams()
        self.output_kwargs = output_kwargs
        self.n_samples = n_samples

        self.calculate_square_gradients = self._compile_sqr_grad_calculation()

    def estimate_total_curvature(self, datasets):
        n_samples = self.n_samples
        diagonal_fishers = []
        for ds in datasets:
            scale = 1 if not n_samples else ds.train_n / n_samples
            n_max = n_samples if n_samples else ds.train_n
            for i, (x, _) in enumerate(ds.iter("train", 1)):
                if i >= n_max:
                    break

                fishers = self.calculate_square_gradients(x.astype(theano.config.floatX))
                if diagonal_fishers:
                    diagonal_fishers = [scale * f + df for f, df in zip(fishers, diagonal_fishers)]
                else:
                    diagonal_fishers = fishers

        return diagonal_fishers

    def _compile_sqr_grad_calculation(self):
        """
        Compiles the theano function for calculating the squared gradient, assuming that only a single datapoint is
        used. If you're passing in a minibatch, you're calculating the square of the total/expected gradient, which
        is unequal to the sum/expectation of the squared gradient.
        """
        loss_layer = self.loss_layer
        output_layer = loss_layer.input_layers[0]
        output = L.get_output(output_layer, **self.output_kwargs)
        if isinstance(loss_layer, L.BinaryLogitsCrossEntropy):
            p = T.nnet.sigmoid(output)
            y = self.rng.multinomial(n=1, pvals=p, dtype="int8")
        elif isinstance(loss_layer, L.CategoricalLogitsCrossEntropy):
            p = T.nnet.softmax(output)
            y = self.rng.multinomial(n=1, pvals=p, dtype="int8")
        else:
            raise ValueError("Impossible!")
        loss = L.get_output(loss_layer, inputs={loss_layer.input_layers[1]: theano.gradient.disconnected_grad(y)},
                            **self.output_kwargs)
        grads = T.grad(loss.sum(), L.get_all_params(loss_layer))
        sqr_grads = list(map(T.sqr, grads))
        return theano.function([self.invar], sqr_grads)


def apply_kf_laplace(loss_layer, qc, gc, mode="prec"):
    """
    Applies a Kronecker factored Laplace approximation (https://openreview.net/pdf?id=Skdvd2xAZ) to the weights of the
    layers feeding into a given loss layer by adding samples from a matrix normal distribution to them.
    Note that you will probably want to regularize the curvature factors by multiplying them by a constant > 0 and
    adding a multiple of the identity > 0, as the Hessian of neural networks is flat in most directions, hence the
    Laplace approximation might not work out of the box (although this depends on your architecture and the size of the
    dataset.

    Parameters
    ----------
    loss_layer : the loss layer of your network (from `lasagne.layers.objectives`; the output layer should also be ok)

    qc : list of first Kronecker factors per layer (i.e. the covariance of the inputs to the layer). Assumed to be
         ordered from input to output.

    gc : list of second Kronecker factors per layer (i.e. the Hessian of the loss w.r.t. the linear outputs). Assumed
         to be in same order as `qc'.

    mode : string indicating whether the `qc' and `gc' are square roots of the covariance or precision.
    """

    curvature_index = -1
    for layer in reversed(L.get_all_layers(loss_layer)):
        if isinstance(layer, L.DenseLayer):
            q, g = qc[curvature_index], gc[curvature_index]
            layer.W_fused = nn.random.matrix_normal(
                layer.W_fused.shape, layer.W_fused, q, g, mode=mode)
            curvature_index -= 1
        elif isinstance(layer, L.Conv2DLayer):
            q, g = qc[curvature_index], gc[curvature_index]
            if layer.fuse_bias:
                layer.W_fused = nn.random.matrix_normal(layer.W_fused.shape, layer.W_fused, q, g, mode=mode)
            else:
                w_shape = layer.get_W_shape()
                w_flat = layer.W.transpose(1, 2, 3, 0).reshape((-1, layer.num_filters))
                w_sample = nn.random.matrix_normal(w_flat.shape, w_flat, q, g, mode=mode)
                layer.W = w_sample.reshape(w_shape[1:] + (layer.num_filters,)).transpose(3, 0, 1, 2)
            curvature_index -= 1
        elif isinstance(layer, (L.BatchNormLayer, L.BatchNormTheanoLayer)):
            beta, gamma = layer.beta, layer.gamma
            beta_cov = qc[curvature_index-1] * T.addbroadcast(gc[curvature_index-1], 0, 1)
            layer.beta = nn.random.multivariate_normal(beta.shape, beta, beta_cov, mode=mode)
            gamma_cov = qc[curvature_index] * T.addbroadcast(gc[curvature_index], 0, 1)
            layer.gamma = nn.random.multivariate_normal(gamma.shape, gamma, gamma_cov, mode=mode)
            curvature_index -= 2
        elif len(layer.get_params()) > 0:
            raise ValueError("Layer '{}' has parameters, however there currently is no implementation for"
                             " applying a kronecker factored laplace approximation to the parameters of a '{}'"
                             " layer".format(layer.name, layer.__class__))
        else:
            # not a parameter layer, do nothing
            pass


def apply_diagonal_laplace(loss_layer, inv_fisher_diagonals):
    """

    Parameters
    ----------
    loss_layer: the loss layer of your network (from `lasagne.layers.objectives`; the output layer should also be ok)

    inv_fisher_diagonals: square root of the inverse diagonal elements of the fisher (of the same shape as the
                          parameters of each layer)
    """
    curvature_index = 0
    for layer in L.get_all_layers(loss_layer):
        if isinstance(layer, L.DenseLayer):
            layer.W_fused += nn.random.normal(layer.W_fused.shape, std=inv_fisher_diagonals[curvature_index])
            curvature_index += 1
        elif isinstance(layer, L.Conv2DLayer):
            if layer.fuse_bias:
                attr = "W_fused"
            else:
                attr = "W"
            w = getattr(layer, attr)
            setattr(layer, attr, nn.random.normal(w.shape, w, inv_fisher_diagonals[curvature_index]))
            curvature_index += 1
        elif isinstance(layer, (L.BatchNormLayer, L.BatchNormTheanoLayer)):
            layer.beta += nn.random.normal(layer.beta.shape, std=inv_fisher_diagonals[curvature_index])
            layer.gamma += nn.random.normal(layer.gamma.shape, std=inv_fisher_diagonals[curvature_index + 1])
            curvature_index += 2
        elif len(layer.get_params()) > 0:
            raise ValueError("Layer '{}' has parameters, however there currently is no implementation for"
                             " applying a diagonal laplace approximation to the parameters of a '{}'"
                             " layer".format(layer.name, layer.__class__))
