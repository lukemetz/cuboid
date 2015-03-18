from theano import tensor as T
import theano
import numpy as np
from numpy.testing import assert_allclose, assert_equal

from cuboid.bricks import Dropout, FilterPool, BatchNormalizationConv, BatchNormalization
from blocks.initialization import Constant

def test_dropout():
    layer = Dropout(p_drop=0.1)
    layer.initialize()

    x = T.matrix()
    y = layer.apply(x)

    _func = theano.function([x], y)
    x_val = np.random.randint(0, 100, (8, 8)).astype(theano.config.floatX)
    v1 = _func(x_val)
    v2 = _func(x_val)

    # these values are different
    diff = np.sum((v2 - v1)**2)
    assert(diff >= .1)

def test_filterpool():
    layer = FilterPool(fn=T.mean, input_dim=(3, 4, 5))
    layer.initialize()

    x = T.tensor4()
    y = layer.apply(x)

    _func = theano.function([x], y)
    x_val = np.zeros((1, 3, 4, 5)).astype(theano.config.floatX)
    x_val[0, 0, 2, 3] = 19.0
    x_val[0, 0, 1, 3] = 21.0
    x_val[0, 1, 2, 3] = 10.0
    x_val[0, 2, 2, 3] = 5.0

    ret = _func(x_val)
    assert_allclose(ret, np.array([[ 2, .5, 0.25]]))

def test_batchnormconv_training():
    layer = BatchNormalizationConv(
            B_init = Constant(0),
            Y_init = Constant(1),
            input_dim = (4, 5, 5))
    layer.initialize()
    x = T.tensor4()

    x_val = np.ones((6, 4, 5, 5), dtype=theano.config.floatX)
    x_val[0,0,0,0] = 10.0

    y = layer.apply(x)
    _func = theano.function([x], y)
    ret = _func(x_val)
    assert_equal(ret.shape, (6, 4, 5, 5))
    assert_allclose(ret[0, 0, 0, 0], 12.20655537)
    assert_allclose(ret[0, 0, 1, 0], -0.08192328)
    assert_allclose(ret[0, 0, 3, 3], -0.08192328)

    assert_allclose(ret[1, 0, 0, 0], -0.08192328)

    assert_allclose(ret[0:6, 1:4, 0:5, 0:5], 0)

def test_batchnormconv_inference():
    layer = BatchNormalizationConv(
            B_init = Constant(0),
            Y_init = Constant(1),
            input_dim = (4, 5, 5))
    layer.population_mean = 2 * np.ones(4)
    layer.population_var = 16 * np.ones(4)
    layer.initialize()
    x = T.tensor4()

    x_val = np.ones((6, 4, 5, 5), dtype=theano.config.floatX)
    x_val[0,0,0,0] = 10.0

    y = layer.apply(x)
    _func = theano.function([x], y)
    ret = _func(x_val)
    assert_equal(ret.shape, (6, 4, 5, 5))
    assert_allclose(ret[0, 0, 0, 0], 2)
    assert_allclose(ret[0, 0, 1, 0], -0.25)
    assert_allclose(ret[0, 0, 3, 3], -0.25)

    assert_allclose(ret[1, 0, 0, 0], -0.25)

    assert_allclose(ret[0:6, 1:4, 0:5, 0:5], -0.25)

def test_batchnorm_training():
    layer = BatchNormalization(
            B_init = Constant(0),
            Y_init = Constant(1),
            input_dim = 5)
    layer.initialize()
    x = T.matrix()

    x_val = np.ones((6, 5), dtype=theano.config.floatX)
    x_val[0,0] = 10.0

    y = layer.apply(x)
    _func = theano.function([x], y)
    ret = _func(x_val)

    assert_allclose(ret[0,0], 2.23606801)
    assert_allclose(ret[1:5, 0], -0.44721359)

    assert_allclose(ret[0:5,1:5], 0)

def test_batchnorm_training():
    layer = BatchNormalization(
            B_init = Constant(0),
            Y_init = Constant(1),
            input_dim = 5)

    layer.population_mean = 2
    layer.population_var = 16

    layer.initialize()
    x = T.matrix()

    x_val = np.ones((6, 5), dtype=theano.config.floatX)
    x_val[0,0] = 10.0

    y = layer.apply(x)
    _func = theano.function([x], y)
    ret = _func(x_val)

    assert_allclose(ret[0,0], 2)
    assert_allclose(ret[1:5, 0], -0.25)

    assert_allclose(ret[0:5,1:5], -0.25)

