from lasagne.serialize import deserialize
from lasagne import layers as L
from lasagne.random import normal
import json
from pprint import pprint


if __name__ == '__main__':
    with open("network.json") as f:
        config = json.load(f)
    cache = deserialize(config, D=784)
    for p in L.get_all_params(config["network"][-1]):
        print(p.name, p.get_value().shape)
    pprint(cache)
    pprint(config)
    for l in config["network"]:
        l.base_pretty_print()
    l = cache[0]["neg_elbo"]
    l_prior = L.InputLayer((None, 64), name="new_z")
    x = normal((200, 784))
    l_sample = cache[0]["z1_sample"]
    replace = {l_sample: l_prior}
    print(L.get_all_layers(l, replace=replace))
    print(l(x=x, z1_sample=l_prior))

