import numpy as np
import theano
import theano.tensor as T
from theano.ifelse import ifelse
from theano.gradient import zero_grad
from collections import OrderedDict
import warnings

from . import random
from . import utils
from . import updates as upd
from .layers.helper import get_all_params, get_output, get_all_layers
from .layers import InputLayer
from .reporting import add_to_report


VARIANTS = ("skfgn-rp", "skfgn-fisher", "kfac*", "kfra")
FUSE_BIAS = False


def get_fuse_bias():
    return FUSE_BIAS


def set_fuse_bias(value):
    global FUSE_BIAS
    FUSE_BIAS = value


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
                 alpha_avg=0.0,
                 alpha_period=10,
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

        :param avg_init_period: int (default: 100)
            An initial period in which the moving average constant for curvature_avg and grad_avg
            grows from 0.1 to the one provided linearly.

        :param grad_avg: float or callable (default: 0.0)
            The averaging parameter for the gradient matrices. 
            If 0, then no averaging will at at each step only the mini-batch matrices will be used.
            If it is a callable it should have the format:
            f(s_t, x_t, t, init_period)

        :param curvature_avg: float or callable (default: 0.0)
            The averaging parameter for the kronecker factors matrices. 
            If 0, then no averaging will at at each step only the mini-batch matrices will be used.
            If it is a callable it should have the format:
            f(s_t, x_t, t, init_period)
        
        :param mirror_avg: float or callable (default: 0.0)
            The averaging parameter for the Polyak averaging on the parameters.
            If 0, then no averaging will be performed and "mirrors" of the parameters
            will not be created. If it is a callable it should have the format:
            f(s_t, x_t, t, init_period)

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
        if variant == "kfac_star":
            variant = "kfac*"
        elif variant not in VARIANTS:
            raise ValueError("Variant must be one of {}".format(str(VARIANTS)))
        self.__variant = variant
        self.random_sampler = random_sampler
        self.norm = norm
        self.tikhonov_damping = tikhonov_damping
        self.rescale_by_gn = rescale_by_gn

        self.avg_init_period = avg_init_period
        self._grad_avg = None
        self.grad_avg = grad_avg
        self._curvature_avg = None
        self.curvature_avg = curvature_avg
        self._mirror_avg = None
        self.mirror_avg = mirror_avg
        self._alpha_avg = None
        self.alpha_avg = alpha_avg
        self.alpha_period = alpha_period

        self.clipping = clipping
        self.learning_rate = learning_rate
        self.update_function = update_function
        # Unpopular options
        self.exact_inversion = exact_inversion
        self.gn_momentum = gn_momentum
        self.tikhonov_period = tikhonov_period
        self.debug = debug
        self._t = None

    @property
    def variant(self):
        return self.__variant

    @variant.setter
    def variant(self, value):
        if value == "kfac_star":
            self.__variant = "kfac*"
        elif value not in VARIANTS:
            raise ValueError("Variant must be one of {}".format(str(VARIANTS)))
        else:
            self.__variant = value

    @property
    def grad_avg(self):
        return self._grad_avg

    @grad_avg.setter
    def grad_avg(self, value):
        if value is None or value == 0:
            avg = None
        elif isinstance(value, (int, float, np.ndarray, T.TensorVariable)):
            def avg(s_t, x_t):
                return upd.ema(value, s_t, x_t, self._t, self.avg_init_period)
        elif callable(value):
            def avg(s_t, x_t):
                return value(s_t, x_t, self._t, self.avg_init_period)
        else:
            raise ValueError("Unrecognized type of grad_avg:" + str(type(value)))
        self._grad_avg = avg

    @property
    def curvature_avg(self):
        return self._curvature_avg

    @curvature_avg.setter
    def curvature_avg(self, value):
        if value is None or value == 0:
            avg = None
        elif isinstance(value, (int, float, np.ndarray, T.TensorVariable)):
            def avg(s_t, x_t):
                return upd.ema(value, s_t, x_t, self._t, self.avg_init_period)
        elif callable(value):
            def avg(s_t, x_t):
                return value(s_t, x_t, self._t, self.avg_init_period)
        else:
            raise ValueError("Unrecognized type of curvature_avg:" + str(type(value)))
        self._curvature_avg = avg

    @property
    def alpha_avg(self):
        return self._alpha_avg

    @alpha_avg.setter
    def alpha_avg(self, value):
        if value is None or value == 0:
            avg = None
        elif isinstance(value, (int, float, np.ndarray, T.TensorVariable)):
            def avg(s_t, x_t):
                return upd.ema(value, s_t, x_t, self._t, self.avg_init_period)
        elif callable(value):
            def avg(s_t, x_t):
                return value(s_t, x_t, self._t, self.avg_init_period)
        else:
            raise ValueError("Unrecognized type of curvature_avg:" + str(type(value)))
        self._alpha_avg = avg

    @property
    def mirror_avg(self):
        return self._mirror_avg

    @mirror_avg.setter
    def mirror_avg(self, value):
        if value is None or value == 0:
            avg = None
        elif isinstance(value, (int, float, np.ndarray, T.TensorVariable)):
            def avg(s_t, x_t):
                return upd.ema(value, s_t, x_t, self._t, self.avg_init_period)
        elif callable(value):
            def avg(s_t, x_t):
                return value(s_t, x_t, self._t, self.avg_init_period)
        else:
            raise ValueError("Unrecognized type of mirror_avg:" + str(type(value)))
        self._mirror_avg = avg

    def __call__(self, loss_layer,
                 l2_reg=None,
                 params=None,
                 true_loss_or_grads=None,
                 treat_as_input=None,
                 return_mirror_map=True,
                 return_loss=True):
        t_was_none = self._t is None
        if t_was_none:
            self._t = theano.shared(np.asarray(0, dtype=np.int64), name="skfgn_t")

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
        grads_map = OrderedDict((p, g) for p, g in zip(params, grads))

        # Calculate all of the Kronecker-Factored matrices
        kf_matrices = self.curvature_propagation(loss_layer, T.constant(1),
                                                 inputs_map, outputs_map,
                                                 treat_as_input)

        # Calculate the SKFGN steps and all extra necessary updates
        initial_steps, extra_updates = self.skfgn_steps(grads_map, kf_matrices)

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
            r, alpha_1, alpha_2 = self.gn_rescale(gauss_newton_product, grads, initial_steps, velocities)
            add_to_report("skfgn_alpha", T.stack(alpha_1, alpha_2))
            add_to_report("skfgn_reduction", r)
            steps = [alpha_1 * step + alpha_2 * v for step, v in zip(initial_steps, velocities)]
        # If rescaling by the true Gauss-Newton
        elif self.rescale_by_gn:
            if self.debug:
                print("Rescaling by the Gauss-Newton matrix.")
            r, alpha = self.gn_rescale(gauss_newton_product, grads, initial_steps)
            # alpha = T.abs_(alpha)
            # Alpha averaging only on certain iterations
            if self.alpha_avg is not None:
                cond1 = T.le(self._t, self.avg_init_period)
                cond2 = T.eq(T.mod(self._t, self.alpha_period), 0)
                cond = T.or_(cond1, cond2)
                alpha_avg = theano.shared(utils.floatX(0.0001), name="skfgn_alpha_avg")
                alpha = self.alpha_avg(alpha_avg, alpha)
                r, alpha = ifelse(cond, (r, alpha), (utils.th_fx(0), alpha_avg))
                extra_updates[alpha_avg] = alpha
            add_to_report("skfgn_alpha", alpha)
            add_to_report("skfgn_reduction", r)
            steps = [step * alpha for step in initial_steps]
        # Else use your standard updates
        else:
            steps = initial_steps

        if self.clipping is not None:
            if self.debug:
                print("Applying clipping.")
            steps = self.clipping(steps, grads)

        # Apply clipping
        grad_norm = T.sqrt(sum(T.sum(T.sqr(g)) for g in grads))
        step_norm = T.sqrt(sum(T.sum(T.sqr(s)) for s in steps))
        mixed_norm = T.sqrt(sum(T.sum(s * g) for s, g in zip(steps, grads)))
        norms = T.stack((grad_norm, step_norm, mixed_norm))
        add_to_report("skfgn::norms", norms)

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
        if self.mirror_avg is not None:
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
                        updates[mirror] = self.mirror_avg(mirror, updates[w])
                        mirror_map[w] = mirror
                        if self.debug:
                            print("Adding", mirror.name, "to updates and mirror map.")
                        continue
        # Just retain the original parameters
        else:
            for w in params:
                mirror_map[w] = w
        # Reset t
        if t_was_none:
            self._t = None

        if return_mirror_map and return_loss:
            return updates, mirror_map, loss
        elif return_mirror_map:
            return updates, mirror_map
        elif return_loss:
            return updates, loss
        else:
            return updates

    def skfgn_steps(self, grads_map, kf_matrices):
        # Create any extra updates
        extra_updates = OrderedDict()
        extra_updates[self._t] = self._t + 1

        # Compute and return the new steps
        steps_map = OrderedDict()
        for mat in kf_matrices:
            # Add the updates for the Kronecker-Factored matrices
            extra_updates.update(mat.get_updates())
            grads = []
            if self.grad_avg is not None:
                for p in mat.params:
                    grad_avg = theano.shared(T.zeros_like(p).eval(), name=p.name + "_avg")
                    grads.append(self.grad_avg(grad_avg, grads_map[p]))
                    extra_updates[grad_avg] = grads[-1]
            else:
                grads = [grads_map[p] for p in mat.params]

            # Multiply the gradients by the inverse Gauss-Newton
            if get_fuse_bias():
                grad = grads[0]
            else:
                grad = mat.layer.params_to_fused(grads)
            joint_step = mat.linear_solve(grad)
            if get_fuse_bias():
                steps = [joint_step]
            else:
                steps = mat.layer.params_from_fused(joint_step, len(mat.params))

            # Exchange the steps in the steps map
            for p, step in zip(mat.params, steps):
                steps_map[p] = step
        steps = []
        for w, g in grads_map.items():
            if steps_map.get(w) is None:
                print("Parameter", w, "was not update with SKFGN, falling back to just gradient")
            steps.append(steps_map.get(w, g))

        return steps, extra_updates

    def curvature_propagation(self, loss_layer, loss_grad=None,
                              inputs_map=None, outputs_map=None,
                              treat_as_input=None):
        if loss_grad is None:
            loss_grad = T.constant(1)
        if inputs_map is None or outputs_map is None:
            _, inputs_map, outputs_map = get_output(loss_layer, get_maps=True)
        # List of all the Kronecker-Factored matrices
        kf_matrices = list()

        # Helper function to create Kronecker-Factored matrices
        if self.variant == "kfra":
            def make_matrix(owning_layer, a_dim, b_dim, a, b, params, name=None):
                name = name + "_" if name else ""
                a_factor = FullMatrix(a_dim, a,
                                      layer=owning_layer, params=params,
                                      k=0.0,
                                      init_k=self.tikhonov_damping,
                                      update_fn=self.curvature_avg,
                                      name=name + "kf_a")
                b_factor = FullMatrix(b_dim, b,
                                      layer=owning_layer, params=params,
                                      k=0.0,
                                      init_k=self.tikhonov_damping,
                                      update_fn=self.curvature_avg,
                                      name=name + "kf_b")
                kf_mat = KroneckerFactoredMatrix((a_factor, b_factor),
                                                 owning_layer,
                                                 k=self.tikhonov_damping)
                kf_matrices.append(kf_mat)
        else:
            def make_matrix(owning_layer, a_dim, b_dim, a_sqrt, b_sqrt, params, name=None):
                name = name + "_" if name else ""
                if a_dim is None:
                    mat = SqrtMatrix(b_dim, b_sqrt,
                                     layer=owning_layer, params=params,
                                     k=0.0,
                                     init_k=self.tikhonov_damping,
                                     update_fn=self.curvature_avg,
                                     name=name)
                elif b_dim is None:
                    mat = SqrtMatrix(a_dim, a_sqrt,
                                     layer=owning_layer, params=params,
                                     k=0.0,
                                     init_k=self.tikhonov_damping,
                                     update_fn=self.curvature_avg,
                                     name=name)
                else:
                    a_factor = SqrtMatrix(a_dim, a_sqrt,
                                          layer=owning_layer, params=params,
                                          k=0.0,
                                          init_k=self.tikhonov_damping,
                                          update_fn=self.curvature_avg,
                                          name=name + "kf_a")
                    b_factor = SqrtMatrix(b_dim, b_sqrt,
                                          layer=owning_layer, params=params,
                                          k=0.0,
                                          init_k=self.tikhonov_damping,
                                          update_fn=self.curvature_avg,
                                          name=name + "kf_b")
                    mat = KroneckerFactoredMatrix((a_factor, b_factor),
                                                  owning_layer,
                                                  name=name,
                                                  k=self.tikhonov_damping)
                kf_matrices.append(mat)

        if self.debug:
            print("Running SKFGN with variant", self.variant)
        # Mapping from layer to propagated curvature
        curvature_map = dict()
        # Mapping from layer to incoming curvature
        input_curvature = dict()
        # Start from the loss layer
        curvature_map[loss_layer] = [loss_grad]
        all_layers = get_all_layers(loss_layer, treat_as_input)
        for layer in reversed(all_layers):
            if self.debug:
                print(layer.name)
            # For input layers nothing to compute
            if isinstance(layer, InputLayer):
                input_curvature[layer] = curvature_map[layer]
                continue

            # Extract the needed information of the current layer
            inputs = inputs_map[layer]
            outputs = outputs_map[layer]
            curvature = curvature_map[layer]

            # Propagate the curvature
            curvature = layer.curvature_propagation(self, inputs, outputs, curvature, make_matrix)
            assert len(curvature) == len(layer.input_layers)

            # Assign the propagated curvature to the corresponding layers
            c = 0
            for i, l in enumerate(layer.input_layers):
                current_curvature = curvature[c: c + l.num_outputs]
                if None in current_curvature and len(current_curvature) > 1:
                    raise NotImplementedError
                if current_curvature[0] is not None:
                    if curvature_map.get(l) is not None:
                        curvature_map[l] = [old + new for old, new in zip(curvature_map[l], current_curvature)]
                    else:
                        curvature_map[l] = current_curvature
                    c += l.num_outputs
                else:
                    if not isinstance(l, InputLayer):
                        warnings.warn("Layer {}({}) has incoming curvature None."
                                      .format(l.name, type(l).__name__))

        return kf_matrices

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
            # gg = gauss_newton_product(steps1, grads)
            # gg = utils.theano_print_values(gg, "gg")
            M = G + kI #+ 1e-20 * gg
            # M = theano_print_vals(M, "M")
            alpha = v / M
            reduction = alpha * v / 2.0
            return reduction, alpha
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
            return reduction, alpha_1, alpha_2


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
        max_norm = primitive_dict.pop("clipping_norm")
        if name == "gauss-newton":
            def clipping(steps, grads):
                return upd.total_norm_constraint(steps, max_norm, grads)
        elif name == "norm":
            def clipping(steps, grads):
                return upd.total_norm_constraint(steps, max_norm)
        elif name == "box":
            def clipping(steps, grads):
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


class PositiveSemiDefiniteMatrix(object):
    def __init__(self, dim,
                 transpose=False,
                 layer=None,
                 params=None,
                 k=1e-3,
                 init_k=None,
                 name=None,
                 update_fn=None):
        self.dim = dim
        # self.transpose = transpose
        self.layer = layer
        self.params = params if params else ()
        self.k = k
        self.init_k = init_k if init_k else k
        self.name = name if name else ""
        self.update_fn = update_fn
        if update_fn is not None:
            self.make_persistent()
        else:
            self.persistent = None
        self.update = None

    def make_persistent(self):
        try:
            value = self.get("raw").eval()
        except :
            value = np.eye(self.dim) * self.init_k
        if self.layer is not None:
            self.persistent = self.layer.add_param(
                utils.floatX(value), (self.dim, self.dim),
                name=self.name + "_psd",
                broadcast_unit_dims=False,
                trainable=False,
                regularizable=False,
                psd=True)
        else:
            self.persistent = theano.shared(utils.floatX(value),
                                            name=self.name + "_psd")

    def get_raw(self):
        raise NotImplementedError

    def get(self, use="updates"):
        if use == "persistent":
            return self.persistent
        elif use == "updates":
            if self.update is not None:
                return self.update
            else:
                return self.get_raw()
        elif use == "raw":
            return self.get_raw()
        else:
            raise ValueError("Unrecognized use=" + use)

    def get_updates(self):
        if self.persistent is None:
            return ()
        else:
            return (self.persistent, self.update),

    def norm(self, norm_fn, use="updates"):
        return norm_fn(self.get(use))

    # def transpose(self):
    #     transposed = copy(self)
    #     transposed.transpose = not self.transpose
    #     return transposed

    def linear_solve(self, v_org, use="updates", scale=None, k_override=None, right_multiply=True):
        assert v_org.ndim <= 2
        mat = self.get(use)
        mat *= scale if scale else 1
        k = k_override if k_override else self.k
        mat += T.eye(self.dim) * k
        if self.dim == 1:
            return v_org / mat[0, 0]
        v = T.reshape(v_org, (-1, 1)) if v_org.ndim < 2 else v_org
        if right_multiply:
            solved = utils.linear_solve(mat, v, "symmetric")
        else:
            solved = utils.linear_solve(mat, v.T, "symmetric").T
        return T.reshape(solved, v_org.shape) if v_org.ndim < 2 else solved

    def cholesky_product(self, v, use="updates", scale=None, k_override=None,
                         transpose=False, right_multiply=True):
        assert v.ndim == 2
        mat = self.get(use)
        mat *= scale if scale else 1
        k = k_override if k_override else self.k
        mat += T.eye(self.dim) * k
        if self.dim == 1:
            return v / T.sqrt(mat[0, 0] + k)
        cholesky = T.slinalg.cholesky(mat)
        if transpose:
            cholesky = cholesky.T
        if right_multiply:
            return T.dot(cholesky, v)
        else:
            return T.dot(v, cholesky)

PSD = PositiveSemiDefiniteMatrix


class FullMatrix(PSD):
    def __init__(self, dim,
                 matrix,
                 transpose=False,
                 layer=None,
                 params=None,
                 k=1e-3,
                 init_k=None,
                 name=None,
                 update_fn=None):
        self.raw_matrix = matrix
        super(FullMatrix, self).__init__(dim, transpose, layer,
                                         params, k, init_k, name,
                                         update_fn)
        if self.update_fn is not None:
            self.update = self.update_fn(self.persistent, self.raw_matrix)

    def get_raw(self):
        return self.raw_matrix


class SqrtMatrix(PSD):
    def __init__(self, dim,
                 sqrt_matrix,
                 n=None,
                 transpose=False,
                 layer=None,
                 params=None,
                 k=1e-3,
                 init_k=None,
                 name=None,
                 update_fn=None):
        self.sqrt_matrix = sqrt_matrix
        self.n = n if n else sqrt_matrix.shape[0]
        super(SqrtMatrix, self).__init__(dim, transpose, layer,
                                         params, k, init_k, name,
                                         update_fn)
        if self.update_fn is not None:
            self.update = self.update_fn(self.persistent, self.get_raw())

    def get_raw(self):
        return T.dot(self.sqrt_matrix.T, self.sqrt_matrix) / utils.th_fx(self.n)


class KroneckerFactoredMatrix(object):
    """
    A Kronecker factored matrix A kron B

    For linear solve it used the identity:
    (A kron B)^-1 vec(V) = vec(B^-1 V A^-1)
    """
    def __init__(self,
                 factors,
                 layer=None,
                 k=1e-3,
                 name=None,
                 norm_fn=utils.mean_trace_norm):
        assert len(factors) > 1
        if len(factors) > 2:
            raise NotImplementedError
        self.factors = factors
        self.params = factors[0].params
        for f in factors[1:]:
            assert all(p1 == p2 for p1, p2 in zip(self.params, f.params))
        self.layer = layer
        self.dim = factors[0].dim * factors[1].dim
        self.k = k
        self.name = name if name else ""
        self.norm_fn = norm_fn

    def get_updates(self):
        if len(self.factors) == 1:
            return self.factors[0].get_updates()
        elif len(self.factors) == 2:
            return self.factors[0].get_updates() + self.factors[1].get_updates()
        elif len(self.factors) == 3:
            return self.factors[0].get_updates() + self.factors[1].get_updates() + \
                   self.factors[2].get_updates()
        elif len(self.factors) == 3:
            return self.factors[0].get_updates() + self.factors[1].get_updates() + \
                   self.factors[2].get_updates() + self.factors[3].get_updates()
        else:
            raise NotImplementedError

    def linear_solve(self, v, use="updates", scale=None, k_override=None, right_multiply=True):
        a, b = self.factors
        k = k_override if k_override else self.k
        scale = scale if scale else 1
        if a.dim == 1 and b.dim == 1:
            mat = scale * a.get(use)[0, 0] * b.get(use)[0, 0]
            return v / (mat + k)
        elif a.dim == 1:
            scale *= a.get(use)[0, 0]
            return b.linear_solve(v, use, scale, k, right_multiply)
        elif b.dim == 1:
            scale *= b.get(use)[0, 0]
            return a.linear_solve(v, use, scale, k, not right_multiply)
        else:
            norm_a = a.norm(self.norm_fn, use)
            norm_b = b.norm(self.norm_fn, use)
            omega = T.sqrt(norm_a / norm_b)
            # relative_norm = T.sqrt(norm_a * norm_b)
            name = self.layer.name + "_" if self.layer else ""
            name += self.name + "_omega"
            add_to_report(name, omega)
            k_override = T.sqrt(k)
            a_scale = T.inv(omega)
            b_scale = omega
            if not right_multiply:
                a, b, a_scale, b_scale = b, a, b_scale, a_scale
            v1 = b.linear_solve(v, use, b_scale, k_override, True)
            v2 = a.linear_solve(v1, use, a_scale, k_override, False)
            return v2

    def cholesky_product(self, v, use="updates", scale=None, k_override=None, right_multiply=True):
        a, b = self.factors
        k = k_override if k_override else self.k
        scale = scale if scale else 1
        if a.dim == 1 and b.dim == 1:
            mat = scale * a.get(use)[0, 0] * b.get(use)[0, 0]
            return v * T.sqrt(mat + k)
        elif a.dim == 1:
            scale *= a.get(use)[0, 0]
            v2 = b.cholesky_product(v, use, scale, k,
                                    transpose=True,
                                    right_multiply=right_multiply)
        elif b.dim == 1:
            scale *= b.get(use)[0, 0]
            v2 = a.cholesky_product(v, use, scale, k,
                                    transpose=False,
                                    right_multiply=not right_multiply)
        else:
            norm_a = a.norm(self.norm_fn, use)
            norm_b = b.norm(self.norm_fn, use)
            omega = T.sqrt(norm_a / norm_b)
            # relative_norm = T.sqrt(norm_a * norm_b)
            # name = self.layer.name + "_" if self.layer else ""
            # name += self.name + "_omega"
            # add_to_report(name, omega)
            k_override = T.sqrt(k)
            a_scale = T.inv(omega)
            b_scale = omega
            if not right_multiply:
                a, b, a_scale, b_scale = b, a, b_scale, a_scale
            v1 = b.cholesky_product(v, use, b_scale, k_override,
                                    transpose=True, right_multiply=True)
            v2 = a.cholesky_product(v1, use, a_scale, k_override,
                                    transpose=False, right_multiply=False)
        return v2
