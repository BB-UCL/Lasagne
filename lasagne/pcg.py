import theano.tensor as T
import theano


def pcg_pack(x_l, r_l, p_l, rtz):
    return x_l + r_l + p_l + [rtz]


def pcg_unpack(params):
    n = len(params) // 3
    return params[:n], params[n:2*n], params[2*n:3*n], params[3*n]


def pcg_init(x0_l, b_l, product_func, precond_func):
    """
    Follows: https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_preconditioned_conjugate_gradient_method
    Initialize a Preconditioned Conjugate Gradient method for solving Ax = b
    :param x0: List of Theano variables representing x0
    :param bs: List of Theano variables representing b
    :param product_func: Calculates Ax
    :param precond_func: Calculates inv(M)x, where M is the preconditioner
    :return: The initial values for r, p, and r^T z
    """
    Ax_l = product_func(x0_l)
    r0_l = [b - ax for b, ax in zip(b_l, Ax_l)]
    p0_l = precond_func(r0_l)
    rtz0 = sum(T.sum(r * p) for r, p in zip(r0_l, p0_l))
    return pcg_pack(x0_l, r0_l, p0_l, rtz0)


def pcg_step(params, product_func, precond_func, stop_criteria=None):
    """
    Follows: https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_preconditioned_conjugate_gradient_method
    Performs a single PCG step
    :param params: A list of parameters packed using pcg_pack
    :param product_func: Calculates Ax
    :param precond_func: Calculates inv(M)x, where M is the preconditioner
    :param stop_criteria: A function, which should return a condition for stopping
    :return: The updated parameters as well as a the condition for stopping
    """
    x_l, r_l, p_l, rtz = pcg_unpack(params)
    Ap_l = product_func(p_l)
    alpha = rtz / sum(T.sum(p * ap) for p, ap in zip(p_l, Ap_l))
    x_new = [x + alpha * p for x, p in zip(x_l, p_l)]
    r_new = [r - alpha * ap for r, ap in zip(r_l, Ap_l)]
    z_new = precond_func(r_new)
    rtz_new = sum(T.sum(z * r) for z, r in zip(z_new, r_new))
    beta = rtz_new / rtz
    p_new = [z + beta * p for z, p in zip(z_new, p_l)]
    params_new = pcg_pack(x_new, r_new, p_new, rtz_new)
    if stop_criteria is not None:
        cond = stop_criteria(params_new)
        return params_new, cond
    else:
        return params_new, T.ge(0.0, 1.0)


def pcg_scan(max_iter, x0_l, b_l, product_func, precond_func, stop_criteria=None):
    init_params = pcg_init(x0_l, b_l, product_func, precond_func)
    n = len(init_params) // 3

    def step(*args):
        params = args[:3*n+1]
        params_new, stop = pcg_step(params, product_func, precond_func, stop_criteria)
        return params_new, theano.scan_module.until(stop)

    result = theano.scan(step,
                         n_steps=max_iter,
                         outputs_info=init_params)
    # Return only the final x list
    return [result[0][i][-1] for i in range(n)], result[0][0].shape[0]

