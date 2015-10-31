from blocks.initialization import Constant
from cuboid.bricks.inception import ConcatBricks
from cuboid.bricks import Convolutional
from numpy.testing import assert_allclose
import theano
import theano.tensor as T
import numpy as np


def test_concat_bricks_conv():
    i1 = Convolutional(input_dim=(4, 16, 16), num_filters=10,
                       filter_size=(1, 1), pad=(0, 0))
    i2 = Convolutional(input_dim=(4, 16, 16), num_filters=12,
                       filter_size=(1, 1), pad=(0, 0))
    brick = ConcatBricks([i1, i2], input_dim=(4, 16, 16))
    brick.weights_init = Constant(0.0)
    brick.biases_init = Constant(0.0)
    brick.initialize()

    x = T.tensor4('input')
    y = brick.apply(x)
    func_ = theano.function([x], [y])

    x_val = np.ones((1, 4, 16, 16), dtype=theano.config.floatX)
    res = func_(x_val)[0]
    assert_allclose(res.shape, (1, 22, 16, 16))
    assert_allclose(brick.get_dim("output"), (22, 16, 16))
