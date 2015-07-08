import numpy
import theano
import shutil

from theano import tensor
from numpy.testing import assert_allclose

from fuel.datasets import IterableDataset
from fuel.streams import DataStream
from blocks.main_loop import MainLoop
from blocks.algorithms import GradientDescent, Scale
from blocks.utils import shared_floatx
from blocks.model import Model

from blocks.extensions import FinishAfter

from cuboid.extensions import ExperimentSaver, UserFunc

floatX = theano.config.floatX

def setup_mainloop(extension):
    """Create a MainLoop, register the given extension, supply it with a
        DataStream and a minimal model/cost to optimize.
    """
    features = [numpy.array(f, dtype=floatX)
                for f in [[1, 2], [3, 4], [5, 6]]]
    dataset = IterableDataset(dict(features=features))
    datastream = DataStream(dataset)

    W = shared_floatx([0, 0], name='W')
    x = tensor.vector('features')
    cost = tensor.sum((x-W)**2)
    cost.name = "cost"

    algorithm = GradientDescent(cost=cost, parameters=[W],
                                step_rule=Scale(1e-3))


    main_loop = MainLoop(
        model=Model(cost), data_stream=datastream,
        algorithm=algorithm,
        extensions=[
            FinishAfter(after_n_epochs=1),
            extension])

    return main_loop

def test_experiment_saver():
    experiment_saver = ExperimentSaver(dest_directory='experimentSaverTest', src_directory='.')
    main_loop = setup_mainloop(experiment_saver)

    # be happy if this doesn't error
    main_loop.run()

    shutil.rmtree('experimentSaverTest')

def test_user_func():
    # make python scope play nice
    called = [False]
    def func(extension):
        called[0] = True

    user_func = UserFunc(after_epoch=True, func=func)
    main_loop = setup_mainloop(user_func)

    main_loop.run()
    assert called[0]==True
