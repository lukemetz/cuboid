from theano import tensor as T
import theano
import numpy as np
from numpy.testing import assert_allclose

from cuboid.bricks import Dropout, FilterPool

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

