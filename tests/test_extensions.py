import numpy
import theano
import shutil
import os

from theano import tensor
from numpy.testing import assert_allclose

from fuel.datasets import IterableDataset
from fuel.streams import DataStream
from blocks.main_loop import MainLoop
from blocks.algorithms import GradientDescent, Scale
from blocks.utils import shared_floatx
from blocks.model import Model

from blocks.extensions import FinishAfter
from blocks.roles import add_role, PARAMETER

from cuboid.extensions import (SourceSaver, UserFunc, LogToFile,
    ExamplesPerSecond, SavePoint, Resume, DirectoryCreator)

floatX = theano.config.floatX

def setup_mainloop(extensions):
    """Create a MainLoop, register the given extension, supply it with a
        DataStream and a minimal model/cost to optimize.
    """
    features = [numpy.array(f, dtype=floatX)
                for f in [[1, 2], [3, 4], [5, 6]]]
    dataset = IterableDataset(dict(features=features))
    datastream = DataStream(dataset)

    W = shared_floatx([0, 0], name='W')
    add_role(W, PARAMETER)
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
            ] + extensions)

    return main_loop

def test_source_saver():
    d = DirectoryCreator(directory="experimentSaverTest")
    experiment_saver = SourceSaver(dest_directory='experimentSaverTest', src_directory='.')
    main_loop = setup_mainloop([d, experiment_saver])

    # be happy if this doesn't error
    main_loop.run()

    shutil.rmtree('experimentSaverTest')

def test_save_point():
    d = DirectoryCreator(directory="savePointTest")
    extension = SavePoint(dest_directory='savePointTest', every_n_batches=3)
    main_loop = setup_mainloop([d, extension])

    # be happy if this doesn't error
    main_loop.run()

    shutil.rmtree('savePointTest')

def test_save_load():
    d = DirectoryCreator(directory="savePointTest")
    extension = SavePoint(dest_directory='savePointTest', every_n_batches=3)
    main_loop = setup_mainloop([d, extension])
    W = main_loop.model.get_parameter_dict().values()[0]
    W.set_value(W.get_value()*0 + 123)
    main_loop.algorithm.step_rule.learning_rate.set_value(0.2)
    main_loop.run()

    # run a second main loop with old values
    extension = Resume('savePointTest', 'iterations_3')
    main_loop = setup_mainloop([extension])
    main_loop.run()

    W = main_loop.model.get_parameter_dict().values()[0]
    value = W.get_value()
    lr = main_loop.algorithm.step_rule.learning_rate.get_value()
    assert_allclose(lr, 0.2)
    assert_allclose(value, [9.22131157, 10.17465496])

    shutil.rmtree('savePointTest')

def test_log_to_file():
    if os.path.exists("_log.csv"):
        os.remove("_log.csv")

    extension = LogToFile("_log.csv")
    main_loop = setup_mainloop([extension])
    main_loop.run()
    assert os.path.exists("_log.csv")
    os.remove("_log.csv")

def test_examples_per_second():
    extension = ExamplesPerSecond()
    main_loop = setup_mainloop([extension])
    main_loop.run()
    assert 'examples_per_second' in main_loop.log.current_row

def test_user_func():
    # make python scope play nice
    called = [False]
    def func(extension):
        called[0] = True

    user_func = UserFunc(after_epoch=True, func=func)
    main_loop = setup_mainloop([user_func])

    main_loop.run()
    assert called[0]==True
