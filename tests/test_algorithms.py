import theano
import theano.tensor as T
from numpy.testing import assert_allclose
from collections import OrderedDict

from blocks.utils import shared_floatx
from cuboid.algorithms import NAG

def test_nag():
    a = shared_floatx([3, 4], name="a")
    cost = (a ** 2).sum()
    steps, updates = NAG(lr=0.01, momentum=0.9).compute_steps(
        OrderedDict([(a, T.grad(cost, a))]))

    f = theano.function([], [steps[a]], updates=updates)

    assert_allclose(f()[0], [ 0.11399999,  0.152     ], rtol=1e-5)
    assert_allclose(f()[0], [ 0.1626    ,  0.21679999], rtol=1e-5)
    assert_allclose(f()[0], [ 0.20633999,  0.27511999], rtol=1e-5)

