from cuboid.bricks.sequence import Conv1D, MaxPooling1D
from numpy.testing import assert_allclose
from blocks.initialization import Constant
from theano import tensor

import numpy as np
import theano

def test_conv1d():
    conv = Conv1D(filter_length=5,
            num_filters=32,
            input_dim=50,
            pad = 2,
            weights_init = Constant(1.0),
            biases_init = Constant(0.2))
    conv.initialize()

    x = tensor.tensor3('input')
    output = conv.apply(x)

    output_val = output.eval({x : np.ones((3, 11, 50), dtype=theano.config.floatX)})

    assert_allclose(output_val.shape, (3,11,32))

def test_maxpool1d():
    maxpool = MaxPooling1D(pooling_length=2)
    maxpool.initialize()

    x = tensor.tensor3('input')
    output = maxpool.apply(x)

    output_val = output.eval({x : np.ones((3, 11, 50), dtype=theano.config.floatX)})
    assert_allclose(output_val.shape, (3, 5, 50))
