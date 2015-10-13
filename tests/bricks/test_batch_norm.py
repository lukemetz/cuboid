from theano import tensor as T
import theano
import numpy as np
from numpy.testing import assert_allclose, assert_equal

from cuboid.bricks import Dropout, FilterPool
from cuboid.bricks.batch_norm import BatchNormalization
from cuboid.bricks import LeakyRectifier, FuncBrick, DefaultsSequence
from cuboid.bricks import Convolutional, Flattener, MaxPooling

from blocks.initialization import Constant
from blocks.bricks import Linear, Rectifier
from fuel.datasets import IterableDataset
from fuel.streams import DataStream
from blocks.graph import ComputationGraph
from blocks.model import Model
from cuboid.bricks.batch_norm import infer_population, get_batchnorm_parameter_values, set_batchnorm_parameter_values

def test_batchnormconv_training():
    layer = BatchNormalization(input_dim = (4, 5, 5))
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

def test_batchnorm_infer():
    layer = BatchNormalization(input_dim = (4, 5, 5))
    layer.accumulate = True
    layer.initialize()
    x = T.tensor4("features")
    x_val = [np.ones((6, 4, 5, 5), dtype=theano.config.floatX) for _ in range(2)]
    x_val[0][0,0,0,0] = 10.0
    x_val[1][0,0,0,0] = -200.0
    y = layer.apply(x)

    dataset = IterableDataset(dict(features=x_val))
    data_stream = DataStream(dataset)
    cg = ComputationGraph([y])

    infer_population(data_stream, cg, 2)

    assert layer.use_population == True
    assert_allclose(layer.u.get_value(), np.array([0.72, 2, 2, 2]))
    assert_allclose(layer.n.get_value(), np.array([2]))

def test_batchnorm_get_set():
    layer = BatchNormalization(input_dim = (4, 5, 5))
    layer.accumulate = True
    layer.initialize()
    x = T.tensor4("features")
    x_val = [np.ones((6, 4, 5, 5), dtype=theano.config.floatX) for _ in range(2)]
    x_val[0][0,0,0,0] = 10.0
    x_val[1][0,0,0,0] = -200.0
    y = layer.apply(x)

    dataset = IterableDataset(dict(features=x_val))
    data_stream = DataStream(dataset)
    cg = Model([y])
    infer_population(data_stream, cg, 2)
    values_dict = get_batchnorm_parameter_values(cg)
    assert len(values_dict.keys()) == 3
    assert_allclose(values_dict['/batchnormalization.n'], np.array([2]))
    values_dict['/batchnormalization.n'] = np.array([5.], dtype="float32")
    set_batchnorm_parameter_values(cg, values_dict)

    assert_allclose(layer.n.get_value(), np.array([5.]))

def test_batchnorm_training():
    layer = BatchNormalization(
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

def test_batchnorm_rolling():
    layer = BatchNormalization(
            input_dim = 5, rolling_accumulate=True)
    layer.initialize()
    x = T.matrix()

    x_val = np.ones((6, 5), dtype=theano.config.floatX)
    x_val[0,0] = 10.0

    y = layer.apply(x)
    cg = ComputationGraph([y])

    _func = cg.get_theano_function()
    for i in range(100):
        ret = _func(x_val)
    u = layer.u.get_value()
    assert_allclose(u[0], 1.58491838)
    assert_allclose(u[1], 0.6339674)

    s = layer.s.get_value()
    assert_allclose(s[0], 7.13214684)
    assert_allclose(s[1], 0.)
