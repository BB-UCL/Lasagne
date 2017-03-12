import numpy as np
import lasagne.layers as L
import lasagne.layers.helper as h
import theano


def test(lcs, precompute, order, learn_init, unroll_scan):
    in_l1 = L.InputLayer((5, 3, 784), name="input")
    in_l2 = L.InputLayer((5, 3, 321), name="input")
    n_in = 3 if order == "TND" else 5
    step_l = lcs((n_in, 1105), 12, name="cell",
                 pre_compute_input=precompute, learn_init=learn_init)
    rec_l = L.RNNLayer((in_l1, in_l2), step_l, name="rec", in_order=order, unroll_scan=unroll_scan)
    r1 = theano.shared(np.random.randn(5, 3, 784).astype(theano.config.floatX))
    r2 = theano.shared(np.random.randn(5, 3, 321).astype(theano.config.floatX))
    out = h.get_output(rec_l, inputs={in_l1: r1, in_l2: r2}).eval()
    print("Predicted:", h.get_output_shape(rec_l))
    print("Actual:   ", out.shape)
    print("Min-max [{:.3f}, {:.3f}]".format(np.min(out), np.max(out)))

if __name__ == '__main__':
    for lcs in (L.StandardStep, L.GRUStep, L.LSTMStep, L.RWAStep):
        for order in ("TND", "NTD"):
            for precompute in (True, False):
                for learn_init in (True, False):
                    for unroll_scan in (True, False):
                        test(lcs, precompute, order, learn_init, unroll_scan)
