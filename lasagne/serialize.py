import regex
from collections import OrderedDict
from . import layers
from . import nonlinearities as nl
from . import init


EXTRA_MODULES = []


def add_extra_module(python_module):
    global EXTRA_MODULES
    EXTRA_MODULES.append(python_module)


def reset_extra_modules():
    global EXTRA_MODULES
    EXTRA_MODULES.clear()


def get_attribute(main_module, name, extra_modules=None):
    global EXTRA_MODULES
    if extra_modules is not None:
        modules = [main_module] + extra_modules + EXTRA_MODULES
    else:
        modules = [main_module] + EXTRA_MODULES

    for mod in modules:
        attr = getattr(mod, name, None)
        if attr is not None:
            return attr
    return None


def add_layer_to_cache(layer, layers_cache):
    layers_cache[layer.name] = layer
    for l in layer.inner_layers.values():
        add_layer_to_cache(l, layers_cache)


def deserialize_layer(kwargs, layers_cache=None, var_cache=None):
    """
    Deserializes a layer.
    :param kwargs:
    :param layers_cache:
    :param var_cache:
    :return:
    """
    clc = get_layer_class(kwargs.pop("layer"))
    deserialize(kwargs, layers_cache, var_cache)
    if layers_cache is not None and layers_cache.get(kwargs.get("name", "")) is not None:
        return layers_cache[kwargs["name"]]
    else:
        instance = clc(**kwargs)
        if layers_cache is not None:
            add_layer_to_cache(instance, layers_cache)
        return instance


def deserialize_nonlinearity(input_dict):
    """
    Deserializes a nonlinearity function.
    :param input_dict:
    :return:
    """
    if isinstance(input_dict, dict):
        type_str = input_dict.pop("name")
    else:
        type_str = input_dict
        input_dict = {}
    if type_str == 'leaky_rectify':
        return nl.LeakyRectify(**input_dict)
    elif type_str == 'leaky_elu':
        return nl.LeakyElu(**input_dict)
    elif type_str == 'scaled_tanh':
        return nl.ScaledTanH(**input_dict)
    elif type_str == "gaussian_parametrization":
        return nl.GaussianParametrization(**input_dict)
    else:
        return get_attribute(nl, type_str)


def deserialize_primitive(value, var_cache=None):
    if var_cache is not None:
        if isinstance(value, (list, tuple)):
            return [var_cache.get(v, v) for v in value]
        else:
            return var_cache.get(value, value)
    else:
        return value


def deserialize(object_dict, layers_cache=None, var_cache=None, **kwargs):
    layers_cache = layers_cache if layers_cache else OrderedDict()
    var_cache = var_cache if var_cache else OrderedDict()
    for k, v in kwargs.items():
        if isinstance(v, layers.Layer):
            layers_cache[k] = v
        else:
            var_cache[k] = v
    for k, v in object_dict.items():
        if k == "network":
            object_dict["network"] = [deserialize_layer(l, layers_cache, var_cache) for l in v]
        elif k in ("layer", "step_layer"):
            object_dict[k] = deserialize_layer(v, layers_cache, var_cache)
        elif k == "incoming":
            if isinstance(v, str):
                object_dict[k] = layers_cache[v]
            elif isinstance(v, (list, tuple)):
                object_dict[k] = [layers_cache[i] for i in v]
            else:
                object_dict[k] = deserialize_layer(v, layers_cache, var_cache)
        elif k == "last_kwargs":
            deserialize(v, layers_cache, var_cache)
        elif k in ("q", "q_logits", "p", "x", "y"):
            if isinstance(v, str):
                object_dict[k] = layers_cache[v]
            else:
                object_dict[k] = deserialize_layer(v, layers_cache, var_cache)
        elif k == "class":
            object_dict[k] = get_layer_class(v)
        elif k == "nonlinearity":
            object_dict[k] = deserialize_nonlinearity(v)
        elif "init" in k:
            cls = get_attribute(init, convert_to_camel(v.pop("name")))
            object_dict[k] = cls(**v)
        elif k in ("name", "num_repeats", "units", "indexes", "num_units", "shape"):
            object_dict[k] = deserialize_primitive(v, var_cache)
        else:
            print("Using primitive default for key `" + k + "`.")
            object_dict[k] = deserialize_primitive(v, var_cache)
    return layers_cache, var_cache


def get_layer_class(class_name):
    class_name = convert_to_camel(class_name)
    cls = get_attribute(layers, class_name)
    if cls is None:
        return get_attribute(layers, class_name + "Layer")
    else:
        return cls


_dims = regex.compile(r'([0-9])(d)')
_under_scorer1 = regex.compile(r'(.)([A-Z0-9][a-z]+)')
_under_scorer2 = regex.compile(r'([a-z0-9])([A-Z0-9])')


def convert_from_camel(name):
    s1 = _under_scorer1.sub(r'\1_\2', name)
    return _under_scorer2.sub(r'\1_\2', s1).lower()


def convert_to_camel(name):
    name = name.lower()
    if '_' in name:
        res = "".join(p.capitalize() for p in name.split('_'))
    elif ' ' in name:
        res = "".join(p.capitalize() for p in name.split('_'))
    elif name[0].islower():
        res = name.capitalize()
    else:
        res = name
    res = res.replace('Lstm', 'LSTM')
    res = res.replace('Gru', 'GRU')
    res = res.replace('Ntm', 'NTM')
    res = res.replace('GaussianKl', 'GaussianKL')
    return _dims.sub(r'\1D', res)
