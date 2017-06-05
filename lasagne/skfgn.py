import numpy as np
import theano
import theano.tensor as T
from theano.gradient import zero_grad
from collections import OrderedDict

from . import random
from . import utils
from . import updates as upd
from .layers.helper import get_all_params, get_output, get_all_layers
from .layers import InputLayer
from .reporting import add_to_report


VARIANTS = ("skfgn-rp", "skfgn-fisher", "kfac*", "kfra")


class Optimizer(object):
    def __init__(self,
                 variant="skfgn-rp",
                 random_sampler=random.uniform_ball,
                 norm=utils.mean_trace_norm,
                 tikhonov_damping=1e-3,
                 rescale_by_gn=True,
                 grad_avg=0.0,
                 curvature_avg=0.0,
                 mirror_avg=0.0,
                 avg_init_period=100,
                 clipping=None,
                 learning_rate=1.0,
                 update_function=upd.sgd,
                 exact_inversion=False,
                 gn_momentum=0,
                 tikhonov_period=0,
                 debug=False):
        """
        Initialization parameters of the optimizer:

        :param variant: str (default: "skfgn-rp")
            One of:
                1. "skfgn-rp" - SKFGN-RP
                2. "skfgn-fisher" - SKFGN-Fisher
                4. "kfac*" - KFAC*
                5. "kfra" - KFRA
                
        :param random_sampler: callable (default: lasagne.random.th_uniform_ball)
            A sampler used for SKFGN-RP

        :param norm: callable (default:  :func:`lasagne.utils.mean_trace_norm` )
            A function which is used to evaluate the norms of the factors Q and G
            when `rescale_by_gn` is True and `exact_inversion` is false

        :param tikhonov_damping: float (default: 1e-3)
            The Tikhonov damping added to the curvature matrix

        :param rescale_by_gn: bool (default: True)
            Given the update step produced whether to use the full Gauss-Newton 
            second order expansion to choose the correct step size.

        :param grad_avg: float (default: 0.0)
            The averaging parameter for the gradient matrices. 
            If 0, then no averaging will at at each step only the mini-batch matrices will be used.
        
        :param curvature_avg: float (default: 0.0)
            The averaging parameter for the kronecker factors matrices. 
            If 0, then no averaging will at at each step only the mini-batch matrices will be used.
        
        :param mirror_avg: float (default: 0.0)
            The averaging parameter for the Polyak averaging on the parameters.
            If 0, then no averaging will be performed and "mirrors" of the parameters
            will not be created.

        :param avg_init_period: int (default: 100)
            An initial period in which the moving average constant for curvature_avg and grad_avg
            grows from 0.1 to the one provided linearly.
            
        :param clipping: callable or None (default: None)
            Any clipping function to be applied. The function should take two arguments - 
                the updates produced and the gradients. 
            
        :param exact_inversion: bool (default: False)
            Whether to perform exact Kronecker inversion using Eigen value decomposition or approximate 
            using linear solvers.
        
        :param gn_momentum: int (default: 0)
            If set to 0 do not apply this.
            The momentum parameter for the two direction momentum in the original KFAC paper.
            Note that if this is set to True, you **MUST** use sgd as the update function
            unless you really really know what you are doing.
            
        :param tikhonov_period: int (default: 0)
            At what period to update the `tihkonov_damping` parameter with the 
            Levenberg-Marquardt heuristic. If 0 then the damping is never updated.
            
        :param: debug: bool (defualt False)
            Whether to print debug info when performing SKFGN
        """
        if variant not in VARIANTS:
            raise ValueError("Variant must be one of {}".format(str(VARIANTS)))
        self.__variant = variant
        self.random_sampler = random_sampler
        self.norm = norm
        self.tikhonov_damping = tikhonov_damping
        self.rescale_by_gn = rescale_by_gn
        self.grad_avg = grad_avg
        self.curvature_avg = curvature_avg
        self.mirror_avg = mirror_avg
        self.avg_init_period = avg_init_period
        self.clipping = clipping
        self.learning_rate = learning_rate
        self.update_function = update_function
        # Unpopular options
        self.exact_inversion = exact_inversion
        self.gn_momentum = gn_momentum
        self.tikhonov_period = tikhonov_period
        self.debug = debug

    @property
    def variant(self):
        return self.__variant

    @variant.setter
    def variant(self, value):
        if value not in VARIANTS:
            raise ValueError("Variant must be one of {}".format(str(VARIANTS)))

    def __call__(self, loss_layer,
                 l2_reg=None,
                 params=None,
                 true_loss_or_grads=None,
                 treat_as_input=None,
                 return_mirror_map=True,
                 return_loss=True):
        # Get parameters
        if params is None:
            params = get_all_params(loss_layer, trainable=True)

        # Get the output of the loss layer as well as the input and output map for all layers
        loss_out, inputs_map, outputs_map = get_output(loss_layer, get_maps=True)

        # Calculate the actual loss
        if true_loss_or_grads is not None:
            if callable(true_loss_or_grads):
                loss = true_loss_or_grads(loss_out)
            else:
                loss = true_loss_or_grads
        else:
            loss = T.mean(loss_out)
        if l2_reg is not None:
            l2_params = get_all_params(loss_layer, regularizable=True)
            loss += l2_reg * sum(T.sum(T.sqr(p)) for p in l2_params)

        # Calculate the actual gradients
        grads = upd.get_or_compute_grads(loss, params)

        # Make a mapping from parameters to their gradients
        steps_map = OrderedDict((p, g) for p, g in zip(params, grads))

        # Calculate the SKFGN steps and all extra necessary updates
        initial_steps, extra_updates, t = self.skfgn(loss_layer, steps_map,
                                                     inputs_map, outputs_map,
                                                     treat_as_input)

        # If using gn_momentum we need to create velocity vectors
        def gauss_newton_product(v1, v2=None):
            return loss_layer.gauss_newton_product(inputs_map, outputs_map,
                                                   params, self.variant, v1, v2)

        if self.gn_momentum:
            def make_velocity(param):
                value = param.get_value(borrow=True)
                # Unfortunately have to initialize to some very small variable
                return theano.shared(np.random.standard_normal(value.shape).astype(value.dtype) / 100,
                                     name=param.name + "_kf_momentum",
                                     broadcastable=param.broadcastable)
            if self.debug:
                print("Applying GN momentum.")
            velocities = [make_velocity(p) for p in params]
            steps = self.gn_rescale(gauss_newton_product, grads, initial_steps, velocities)
        # If rescaling by the true Gauss-Newton
        elif self.rescale_by_gn:
            if self.debug:
                print("Rescaling by the Gauss-Newton matrix.")
            steps = self.gn_rescale(gauss_newton_product, grads, initial_steps)
        # Else use your standard updates
        else:
            steps = initial_steps

        # Apply clipping
        if self.clipping is not None:
            if self.debug:
                print("Applying clipping.")
            steps = self.clipping(steps, grads)

        # Generate final updates
        updates = self.update_function(steps, params, self.learning_rate)

        # Add all extra updates from SKFGN
        if self.debug:
            print("Standard parameters' updates:")
            for p, v in updates.items():
                print(p.name, p.get_value().shape, ":=", v)
            print("Extra updates:")
            for u, v in extra_updates.items():
                print(u.name, u.get_value().shape, ":=", v)
        # updates.update(extra_updates)
        for u, v in extra_updates.items():
            assert updates.get(u) is None
            updates[u] = v

        # Create mirroring map useful for evaluating validation etc..
        mirror_map = dict()
        # For each parameter create a smoothed mirroring parameter
        if self.mirror_avg > 0:
            gamma = self.get_gamma(self.mirror_avg, t)
            # gamma = utils.theano_print_values(gamma, "gamma")
            all_layers = get_all_layers(loss_layer)
            for w in params:
                for l in all_layers:
                    if w in l.params:
                        name = w.name.split(utils.SCOPE_DELIMITER)[-1]
                        shape = w.shape.eval()
                        mirror = l.add_param(w.get_value(),
                                             shape,
                                             name=name + "_mirror",
                                             broadcast_unit_dims=False,
                                             trainable=False,
                                             regularizable=False,
                                             mirror=True)
                        updates[mirror] = gamma * mirror + (1 - gamma) * updates[w]
                        mirror_map[w] = mirror
                        if self.debug:
                            print("Adding", mirror.name, "to updates and mirror map.")
                        continue
        # Just retain the original parameters
        else:
            for w in params:
                mirror_map[w] = w

        if return_mirror_map and return_loss:
            return updates, mirror_map, loss
        elif return_mirror_map:
            return updates, mirror_map
        elif return_loss:
            return updates, loss
        else:
            return updates

    def skfgn(self, loss_layer, grads_map, inputs_map, outputs_map,
              loss_grad=None, treat_as_input=None):
        loss_grad = 1 if loss_grad is None else loss_grad
        curvature_map = dict()
        input_curvature = dict()

        steps_map = OrderedDict()
        extra_updates = OrderedDict()
        t = theano.shared(np.asarray(0, dtype=np.int64), name="skfgn_t")
        extra_updates[t] = t + 1
        t = utils.th_fx(extra_updates[t])

        def kronecker_inversion(owning_layer, params, q, g):
            if isinstance(params, (list, tuple)):
                grads = [grads_map[p] for p in params]
            else:
                grads = grads_map[params]
            self.kronecker_inversion(owning_layer, params, q, g, grads,
                                     steps_map,  t, extra_updates)

        # Start from the loss layer
        curvature_map[loss_layer] = [loss_grad]
        all_layers = get_all_layers(loss_layer, treat_as_input)
        if self.debug:
            print("Performing SKFGN backprop")
        for layer in reversed(all_layers):
            if self.debug:
                print(layer.name)
            if isinstance(layer, InputLayer):
                input_curvature[layer] = curvature_map[layer]
                continue

            # Extract the needed information of the current layer
            inputs = inputs_map[layer]
            outputs = outputs_map[layer]
            curvature = curvature_map[layer]
            curvature = layer.skfgn(self, inputs, outputs, curvature, kronecker_inversion)
            assert len(curvature) == len(layer.input_layers)

            # Assign the output curvature to the corresponding layers
            c = 0
            for i, l in enumerate(layer.input_layers):
                current_curvature = curvature[c: c + l.num_outputs]
                if curvature_map.get(l) is not None:
                    curvature_map[l] = [old + new for old, new in zip(curvature_map[l], current_curvature)]
                else:
                    curvature_map[l] = current_curvature

        steps = []
        for w, g in grads_map.items():
            steps.append(steps_map.get(w, g))
            if steps_map.get(w) is None:
                print("Parameter", w, "was not update with SKFGN, falling back to just gradient")
        # extra_updates = OrderedDict()
        # steps = []
        #
        # t = utils.th_fx(extra_updates[t])
        # for w, grad in grads_map.items():
        #     # Extract all layers owning the parameter
        #     owning_layers = []
        #     for l in all_layers:
        #         if w in l.params:
        #             owning_layers.append(l)
        #             break
        #     # Calculate the steps for each parameter
        #     step = self.kronecker_inversion(owning_layers, w, q_map[w], g_map[w], grad, t, extra_updates)
        #     steps.append(step)

        return steps, extra_updates, t

    def kronecker_inversion(self, layer, w, q, g, grad, steps_map, t, updates):
        b = None
        name = w.name.split(utils.SCOPE_DELIMITER)[-1]
        if isinstance(w, (tuple, list)):
            assert len(w) == 2
            assert len(grad) == 2
            w, b = w
            grad = T.concatenate((grad[0], grad[1].reshape((1, -1))), axis=0)
            name += "_fused"
        if w.ndim != 2:
            raise ValueError("Currently SKFGN is implemented only for matrix parameters."
                             "Try absorbing the bias in the weight matrix via 'fused_bias=True'")
        q_dim, g_dim = w.shape.eval()
        if b is not None:
            q_dim += 1

        # Curvature smoothing/momentum
        if self.curvature_avg > 0:
            q_avg = layer.add_param(utils.floatX(np.eye(q_dim) * self.tikhonov_damping),
                                    (q_dim, q_dim),
                                    name=name + "_q",
                                    broadcast_unit_dims=False,
                                    trainable=False,
                                    regularizable=False,
                                    skfgn=True)
            g_avg = layer.add_param(utils.floatX(np.eye(g_dim) * self.tikhonov_damping),
                                    (g_dim, g_dim),
                                    name=name + "_g",
                                    broadcast_unit_dims=False,
                                    trainable=False,
                                    regularizable=False,
                                    skfgn=True)
            # Linearly increasing moving average parameter for the first `avg_init_period` iterations
            gamma = self.get_gamma(self.curvature_avg, t)
            # The smoothed curvature matrices
            q = gamma * q_avg + (1 - gamma) * q
            g = gamma * g_avg + (1 - gamma) * g
            # Add smoothed curvature matrices to updates
            updates[q_avg] = q
            updates[g_avg] = g

        # Gradient smoothing/momentum
        if self.grad_avg:
            grad_avg = layer.add_param(utils.floatX(np.zeros((q_dim, g_dim))),
                                       (q_dim, g_dim),
                                       name=name + "_grad",
                                       broadcast_unit_dims=False,
                                       trainable=False,
                                       regularizable=False,
                                       stats=True)
            gamma = self.get_gamma(self.grad_avg, t)
            grad = gamma * grad_avg + (1 - gamma) * grad
            updates[grad_avg] = grad
        if q_dim == 0 and g_dim == 0:
            step = grad / (q + g + self.tikhonov_damping)
        elif q_dim == 0:
            g += T.eye(g_dim) * self.tikhonov_damping
            step = utils.linear_solve(g, grad.T, "symmetric").T / q
        elif g_dim == 0:
            q += T.eye(q_dim) * self.tikhonov_damping
            step = utils.linear_solve(q, grad, "symmetric") / g
        elif self.exact_inversion:
            raise NotImplementedError
        else:
            omega = T.sqrt(self.norm(q) / self.norm(g))
            add_to_report(w.name + "_omega", omega)
            q += T.eye(q_dim) * T.sqrt(self.tikhonov_damping) * omega
            g += T.eye(g_dim) * T.sqrt(self.tikhonov_damping) / omega
            step = utils.linear_solve(q, grad, "symmetric")
            step = utils.linear_solve(g, step.T, "symmetric").T
        if b is None:
            steps_map[w] = step
        else:
            steps_map[w] = step[:-1]
            steps_map[b] = step[-1]

    def gn_rescale(self, gauss_newton_product, grads, steps1, steps2=None):
        # The second order Taylor expansion of f(x) gives us:
        # $$ f(x - D \alpha) \approx f(x) - \nabla^T D \alpha + \frac{1}{2} \alpha^T D^T (G + kI) D \alpha $$
        # Let v = D^T \nabla
        # Let M = D^T (G + kI) D
        # $$ f(x - D \alpha) = f(x) - v^T \alpha + \frac{1}{2} \alpha^T M \alpha $$
        # The solution for is $$ \alpha^* = M^{-1}v $$
        # The value of the quadratic approximation at the optimal value is
        # $$ f(x - D \alpha^*) \approx f(x) - \frac{1}{2}v^T M^{-1} v $$
        # So
        # $$ f(x) - f(x - D \alpha^*) \approx \frac{1}{2}v^T \alpha^*  $$
        # Note that we intentionally consider $$ x - D \alpha $$, such that this can be plugged in
        # to standard descend algorithms

        steps1 = [zero_grad(step) for step in steps1]
        if steps2 is None:
            # In this case we have that $$D$$ is a single column vector and $$\alpha$$ is a scalar.
            # This means that $$M$$ is also a scalar as well as $$v$$.
            v = sum(T.sum(d * g) for d, g in zip(steps1, grads))
            G = gauss_newton_product(steps1)
            kI = sum(T.sum(T.sqr(d)) for d in steps1) * self.tikhonov_damping
            M = G + kI
            # M = theano_print_vals(M, "M")
            alpha = v / M
            add_to_report("kf_alpha", alpha)
            # alpha = theano_print_vals(alpha, "alpha")
            reduction = alpha * v / 2.0
            add_to_report("kf_reduction", reduction)
            return [step * alpha for step in steps1]
        else:
            steps2 = [zero_grad(step) for step in steps2]
            # In this case we have that $$D$$ has two columns vector and $$\alpha$$ is a 2D vector.
            # This means that $$M$$ is a 2x2 matrix and $$v$$ is a 2D vector as well.
            # Note that also $$M$$ is symmetric. Thus let $$M=[M_1, M_{off}; M_{off}, M_2]$$.
            # Then $$det(M) = M_1*M_2 - M_{off}^2$$ and $$M^{-1} = det(M)^{-1} [M_2, - M_{off}; - M_{off}, M_1]$$
            # Thus $$\alpha^* = det(M)^{-1} [M_2 v_1 - M_{off} v_2; M_1 v_2 - M_{off} v_1$$
            # Also $$v^T \alpha^* = det(M)^{-1}(M_2 v_1^2 - 2 * M_{off} v_2 v_1 + M_1 v_2^2)
            # steps2 = [step + 1e-3 for step in steps1]
            G_1 = gauss_newton_product(steps1)
            kI_1 = sum(T.sum(T.sqr(d)) for d in steps1) * self.tikhonov_damping
            M_1 = G_1 + kI_1
            # M_1 = theano_print_vals(M_1, "M_1")

            G_2 = gauss_newton_product(steps2)
            kI_2 = sum(T.sum(T.sqr(d)) for d in steps2) * self.tikhonov_damping
            M_2 = G_2 + kI_2
            # M_2 = theano_print_vals(M_2, "M_2")

            G_off = gauss_newton_product(steps1, steps2)
            kI_off = sum(T.sum(d1 * d2) for d1, d2 in zip(steps1, steps2)) * self.tikhonov_damping
            M_off = G_off + kI_off
            # M_off = theano_print_vals(M_off, "M_off")
            common = T.max(T.abs_(T.stack((M_1, M_2, M_off))))
            M_1 /= common
            M_2 /= common
            M_off /= common

            detM = M_1 * M_2 - T.sqr(M_off)
            # detM = theano_print_vals(detM, "detM")
            v_1 = sum(T.sum(d * g) / common for d, g in zip(steps1, grads))
            v_2 = sum(T.sum(d * g) / common for d, g in zip(steps2, grads))

            alpha_1 = (M_2 * v_1 - M_off * v_2) / detM
            alpha_2 = (M_1 * v_2 - M_off * v_1) / detM
            # alpha_1 = theano_print_vals(alpha_1, "alpha_1")
            # alpha_2 = theano_print_vals(alpha_2, "alpha_2")

            reduction = (alpha_1 * v_1 + alpha_2 * v_2) / 2.0
            add_to_report("kf_alpha", T.stack((alpha_1, alpha_2)))
            add_to_report("kf_reduction", reduction)
            return [step_1 * alpha_1 + step_2 * alpha_2 for step_1, step_2 in zip(steps1, steps2)]

    def get_gamma(self, mu, t):
        c = mu / (self.avg_init_period - 1)
        return T.minimum(mu, t * (self.curvature_avg - c) / self.avg_init_period + c)


def optimizer_from_dict(primitive_dict):
    """
    Takes a dictionary of primitive values (e.g. str, int, floats) and converts it 
    to corresponding arguments for and `Optimizer`. Any missing arguments will be
    filled with the default values. The converted keys are:
        
        random_sampler - str
            Constructs the function for sampling based on the string.
            Can be one four possible choices:
                "normal" - samples from the standard normal distribution.
                "binary" - samples uniform binary variables with values +/- 1.
                "uniform" - samples uniform variables from [-sqrt(3), sqrt(3)]
                "ball" - samples from the D-dimensional uniform ball with.
                    radius sqrt(D+2)
                "index" - samples uniformly an index from [0, D-1] and returns
                    a one-hot encoding of the samples.
        
        norm - str
            Constructs the function for calculating the matrix norm.
            Can be one of two values:
                "frobenius" - Frobenius norm 
                "trace" - Trace norm
            Note that in all cases the actual function will be either the 
            Mean Frobenius norm or Mean Trace norm. This is because 
            ||kron(A, I)/kron(B, I)|| = ||A||* / ||B||*
            Where ||.|| is the usual norm, while ||.||* is the mean norm. 
        
        clipping - str 
            Constructs the function for clipping the updates of the algorithm.
            Can be one of three values:
                "gauss-newton" - clips based on the approximate induced 
                    Gauss-Newton norm.
                "norm" - clips if the total norm of the step exceeds a value.
                "box" - clips each element individually between [-v, v]
        
        update_function - str 
            Constructs the function for updating the parameters.
            Can be one of values:
                "sgd" - Stochastic Gradient Descend
                "momentum" - SGD with momentum
                "nag" - Nesterov's Accelerated Gradient
                "adam" - Adam
            If "update_kwargs" is also present in the dictionary they will be 
            removed and passed to the underlying update function.
            
        burnout - int 
            If present and higher than 0 than will apply burnout to the 
            update function for the parameters.

    Parameters
    ----------

    primitive_dict: dict
        Dictionary with primitive variables

    Returns
    -------
    An `Optimizer` constructed from the dictionary.
    """
    if primitive_dict.get("random_sampler") is not None:
        name = primitive_dict["random_sampler"]
        if name == "normal":
            def sample(shapes):
                return random.normal(shapes)
        elif name == "binary":
            def sample(shapes):
                return random.binary(shapes, v0=-1)
        elif name == "uniform":
            sqrt3 = utils.th_fx(T.constant(np.sqrt(3)))

            def sample(shapes):
                return random.uniform(shapes, min=-sqrt3, max=sqrt3)
        elif name == "ball":
            def sample(shapes):
                return random.uniform_ball(shapes)
        elif name == "index":
            def sample(shapes):
                samples = random.multinomial(shapes, dtype=theano.config.floatX)
                if isinstance(samples, (list, tuple)):
                    return [s * T.sqrt(utils.th_fx(shapes[0][1])) for s in samples]
                else:
                    return samples * T.sqrt(utils.th_fx(shapes[1]))
        else:
            raise ValueError("Unrecognized 'random_sampler'=", name)
        primitive_dict["random_sampler"] = sample

    if primitive_dict.get("norm") is not None:
        name = primitive_dict["norm"]
        if name == "frobenius":
            primitive_dict["norm"] = utils.mean_frobenius_norm
        elif name == "trace":
            primitive_dict["norm"] = utils.mean_trace_norm
        else:
            raise ValueError("Unrecognized 'norm'=", name)

    if primitive_dict.get("clipping") is not None:
        name = primitive_dict["clipping"]
        if name == "gauss-newton":
            def clipping(steps, grads, max_norm):
                return upd.total_norm_constraint(steps, max_norm, grads)
        elif name == "norm":
            def clipping(steps, grads, max_norm):
                return upd.total_norm_constraint(steps, max_norm)
        elif name == "box":
            def clipping(steps, grads, max_norm):
                return T.clip(steps, min=-max_norm, max=max_norm)
        else:
            raise ValueError("Unrecognized 'clipping'=", name)
        primitive_dict["clipping"] = clipping

    if primitive_dict.get("update_function") is not None:
        if primitive_dict.get("update_kwargs"):
            kwargs = primitive_dict.pop("update_kwargs")
        else:
            kwargs = dict()
        name = primitive_dict["update_function"]
        if name == "sgd":
            def update_function(steps, params, learning_rate):
                return upd.sgd(steps, params, learning_rate, **kwargs)
        elif name == "momentum":
            def update_function(steps, params, learning_rate):
                return upd.momentum(steps, params, learning_rate, **kwargs)
        elif name == "nag":
            def update_function(steps, params, learning_rate):
                return upd.nesterov_momentum(steps, params, learning_rate, **kwargs)
        elif name == "adam":
            def update_function(steps, params, learning_rate):
                return upd.adam(steps, params, learning_rate, **kwargs)
        else:
            raise ValueError("Unrecognized 'update_function'=", name)
        primitive_dict["update_function"] = update_function
    else:
        primitive_dict["update_function"] = upd.sgd

    if primitive_dict.get("burnout") is not None:
        burnout = primitive_dict.pop("burnout")
        if burnout > 0:
            primitive_dict["update_function"] = upd.wrap_with_burnout(primitive_dict["update_function"], burnout)

    return Optimizer(**primitive_dict)
