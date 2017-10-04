from .base import Layer
from .dense import DenseLayer


class SequenceRepeatLayer(Layer):
    def __init__(self, incoming,
                 num_repeats,
                 layer_class,
                 varying_kwargs,
                 last_kwargs=None,
                 name=None,
                 prefix=None,
                 **kwargs):
        for k, v in varying_kwargs.items():
            assert len(v) == num_repeats
        inner_layers = list()
        layer = incoming
        for i in range(num_repeats):
            this_kwargs = dict((k, v[i]) for k, v in varying_kwargs.items())
            this_kwargs.update(kwargs)
            layer = layer_class(layer,
                                name=str(i+1),
                                prefix=prefix,
                                **this_kwargs)
            inner_layers.append(layer)
        if last_kwargs is not None:
            this_kwargs = dict(**kwargs)
            this_kwargs.update(last_kwargs)
            layer = layer_class(layer,
                                name=str(num_repeats + 1),
                                prefix=prefix,
                                **this_kwargs)
            inner_layers.append(layer)
        super(SequenceRepeatLayer, self).__init__(
            incoming, inner_layers=inner_layers,
            name=name, prefix=prefix, max_inputs=10,
            **kwargs)

    def get_output_shapes_for(self, input_shapes):
        for l in self.inner_iter:
            input_shapes = l.get_output_shapes_for(input_shapes)
        return input_shapes

    def get_outputs_for(self, inputs, **kwargs):
        for l in self.inner_iter:
            inputs = l.get_outputs_for(inputs, **kwargs)
        return inputs

    def serialize(self):
        kwargs = self.base_serialize()
        return kwargs


class DenseStackLayer(SequenceRepeatLayer):
    def __init__(self, incoming, units, num_repeats=None, *args, **kwargs):
        if num_repeats is None:
            assert isinstance(units, (tuple, list))
            num_repeats = len(units)
        elif isinstance(units, int):
            units = [units for _ in range(num_repeats)]
        varying_kwargs = dict(num_units=units)
        super(DenseStackLayer, self).__init__(
            incoming, num_repeats, DenseLayer, varying_kwargs, *args, **kwargs)
