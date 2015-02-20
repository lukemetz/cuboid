from theano import tensor as T
import theano
import numpy as np

from cuboid.bricks import Dropout

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
