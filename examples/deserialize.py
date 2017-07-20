from lasagne.serialize import deserialize
from lasagne import layers as L
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

