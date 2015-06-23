import theano
from theano import tensor as T
import numpy as np
from numpy.testing import assert_allclose

from blocks.bricks import Linear
from blocks.initialization import Constant
from cuboid.bricks.domain_unsupervised import gradient_reversal, BatchwiseSplit, DomainUnsupervisedCost

def test_gradient_reversal_op():
    x = T.matrix("x")
    z = x * x
    z = gradient_reversal(z)
    dz = T.grad(T.sum(z), wrt=x)
    func = theano.function([x], [dz])
    neg_grad = func(np.ones((3, 1), dtype=theano.config.floatX))
    assert_allclose(neg_grad, -2*np.ones_like(z))

def test_batchwise_split():
    x = T.matrix("x")
    split = BatchwiseSplit(10)
    u,v = split.apply(x)
    func = theano.function([x], [u, v])
    x_val = np.zeros((14, 3), dtype=theano.config.floatX)
    x_val[10:] = 1.0

    u,v = func(x_val)

    assert_allclose(u, np.zeros((10, 3)))
    assert_allclose(v, np.ones((4, 3)))

def test_domain_unsupervised_cost():
    x = T.matrix("stacked features")
    transform = Linear(weights_init=Constant(1.0), use_bias=False, input_dim=5, output_dim=2)
    domain_unsup = DomainUnsupervisedCost(10, transform)
    domain_unsup.initialize()

    cost = domain_unsup.apply(x)

    func = theano.function([x], [cost])
    out = func(np.ones((14, 5), dtype=theano.config.floatX))
    assert_allclose(out, -3.218876)
