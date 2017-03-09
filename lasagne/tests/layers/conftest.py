from mock import Mock
import pytest


@pytest.fixture
def dummy_input_layer():
    from lasagne.layers.input import InputLayer
    input_layer = InputLayer((2, 3, 4))
    mock = Mock(input_layer)
    mock.shape = input_layer.shape
    mock.input_var = input_layer.input_var
    mock.output_shapes = input_layer.output_shapes
    return mock
