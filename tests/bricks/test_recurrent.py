from cuboid.bricks.recurrent import GatedRecurrentFork
from blocks.initialization import Constant

from theano import tensor
import numpy as np
from numpy.testing import assert_allclose
import theano

def test_gated_recurrent_fork_specific():
    brick = GatedRecurrentFork(input_dim=12, hidden_dim=9)
    brick.inputs_weights_init = Constant(0.0)
    brick.inputs_biases_init = Constant(1.0)

    brick.reset_weights_init = Constant(2.0)
    brick.reset_biases_init = Constant(0.0)

    brick.update_weights_init = Constant(1.0)
    brick.update_biases_init = Constant(0.0)

    brick.initialize()

    x = tensor.matrix("input")
    result_dict = brick.apply(x, as_dict=True)

    inputs = result_dict['inputs']
    gate_inputs = result_dict['gate_inputs']

    _func = theano.function([x], [inputs, gate_inputs])
    x = np.zeros((1, 12), dtype="float32")
    inputs, gate_inputs = _func(x)
    assert_allclose(inputs, np.ones((1, 9)))

    x = np.ones((1, 12), dtype="float32")
    inputs, gate_inputs = _func(x)

    assert_allclose(gate_inputs[:, :9], np.ones((1, 9))*12)
    assert_allclose(gate_inputs[:, 9:], np.ones((1, 9))*24)
