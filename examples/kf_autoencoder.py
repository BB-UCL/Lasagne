import lasagne.layers as L
import numpy as np
from lasagne.skfgn import optimizer_from_dict
import theano.tensor as T
from lasagne.nonlinearities import tanh, identity
import lasagne.updates as upd
import theano
from bb_datasets import get_dataset
import time


def autoencoder(arch, binary=True, nonl=tanh):
    # Input
    # x_in = theano.shared(np.random.randn(100, arch[0])).astype(theano.config.floatX)
    l_in = L.InputLayer((None, arch[0]), name="x")
    # Encoder
    layer = l_in
    for i in range(1, len(arch)):
        layer = L.DenseLayer(layer, arch[i], nonlinearity=nonl,
                             name="encode_" + str(i))
        # layer = L.NonlinearityLayer(layer, nonl, name="encode_" + str(i) + "_a")
    # Decoder
    for i in reversed(range(1, len(arch) - 1)):
        layer = L.DenseLayer(layer, arch[i], nonlinearity=nonl,
                             name="decode_" + str(i))
        # layer = L.NonlinearityLayer(layer, nonl, name="decode_" + str(i) + "_a")
    # P(x|z)
    layer = L.DenseLayer(layer, arch[0], nonlinearity=identity, name="p")
    if binary:
        l_loss = L.BinaryLogitsCrossEntropy(layer, l_in,
                                            name="crossentropy")
    else:
        l_loss = L.SquaredLoss(layer, l_in,
                               name="squareloss")
    return [l_in.input_var], l_loss


def convae(arch, binary=True, nonl=tanh):
    # Input
    # x_in = theano.shared(np.random.rand(100, 784)).astype(theano.config.floatX)
    l_in = L.InputLayer((None, 784), name="x")
    layer = L.reshape(l_in, (-1, 1, 28, 28))
    # Encoder
    for i in range(1, len(arch)):
        layer = L.Conv2DLayer(layer, arch[i], (3, 3), pad="same",
                              nonlinearity=nonl, name="encode_" + str(i))
        layer = L.MaxPool2DLayer(layer, (2, 2))
        # layer = L.NonlinearityLayer(layer, nonl, name="encode_" + str(i) + "_a")
    # Decoder
    for i in reversed(range(1, len(arch) - 1)):
        layer = L.DenseLayer(layer, arch[i], nonlinearity=nonl,
                             name="decode_" + str(i))
        # layer = L.NonlinearityLayer(layer, nonl, name="decode_" + str(i) + "_a")
    # P(x|z)
    layer = L.DenseLayer(layer, arch[0], nonlinearity=identity, name="p")
    if binary:
        l_loss = L.BinaryLogitsCrossEntropy(layer, l_in,
                                            name="crossentropy")
    else:
        l_loss = L.SquaredLoss(layer, l_in,
                               name="squareloss")
    return [l_in.input_var], l_loss


def main(dataset="mnist", batch_size=1000, epochs=20, seed=413):
    print("Data-set:", dataset)
    print("Batch size:", batch_size)
    print("Epochs:", epochs)
    if seed is not None:
        np.random.seed(seed)
    # Make model
    if dataset == "faces":
        input_dim = 625
        output_dim = None
        binary = False # if model == "autoencoder" else True
    elif dataset == "curves":
        input_dim = 784
        output_dim = None
        binary = True
    elif dataset == "mnist":
        input_dim = 784
        output_dim = 10
        binary = True
    else:
        raise ValueError("Unrecognized dataset:", dataset)
    arch = (input_dim, 64, 64, 128, 128)
    in_vars, l_loss = autoencoder(arch, binary=binary)
    # in_vars, l_loss = convae(arch, binary=binary)
    optim_args = {"variant": "kfra",
                  "curvature_avg": 0.9,
                  "mirror_avg": 0.9,
                  "random_sampler": "index"}
    optimizer = optimizer_from_dict(optim_args)
    updates, mirror_map, loss = optimizer(l_loss)
    train_f = theano.function(in_vars, loss, updates=updates)
    # loss = T.mean(L.get_output(l_loss))
    # params = L.get_all_params(l_loss)
    # updates = upd.cocob(loss, params, use_sigmoid=True)
    # updates = upd.adam(loss, params, learning_rate=1e-3)
    # train_f = theano.function(in_vars, loss, updates=updates)
    # Prepare data
    data = get_dataset(dataset)
    data.load()
    data_size = data.data["train"][0].shape[0]
    batches = int(np.ceil(float(data_size) / float(batch_size)))
    print("Number of batches:", batches)
    results = np.zeros((batches * epochs, 3))
    i = 0

    def data_transform(x, y):
        return x,

    for e in range(epochs):
        for x, y in data.iter("train", batch_size):
            start_time = time.time()
            # train_f()
            results[i, 1] = train_f(*data_transform(x, y))
            results[i, 0] = time.time() - start_time
            print("[{:.2f}s][{}]Loss:".format(results[i, 0], i), results[i, 1])
            i += 1
    print("Run complete")


if __name__ == '__main__':
    main()
