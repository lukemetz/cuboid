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
from cuboid.extensions.variable_modifiers import (ExponentialDecay, StepDecay, 
    InverseDecay, PolynomialDecay, 
    SigmoidDecay, LinearDecay, ComposeDecay)
from blocks.roles import add_role, PARAMETER

from cuboid.extensions import (SourceSaver, UserFunc, LogToFile,
    ExamplesPerSecond, SavePoint, Resume, DirectoryCreator, Profile)
from cuboid.extensions.monitoring import PerClassAccuracyMonitor, AUCMonitor

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
    assert_allclose(value, [29.273739,  30.059032])

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
    assert 'examples_per_second_rolling' in main_loop.log.current_row
    assert 'examples_seen' in main_loop.log.current_row
    assert 'examples_per_second_total' in main_loop.log.current_row

def test_user_func():
    # make python scope play nice
    called = [False]
    def func(extension):
        called[0] = True

    user_func = UserFunc(after_epoch=True, func=func)
    main_loop = setup_mainloop([user_func])

    main_loop.run()
    assert called[0]==True

def test_profile():
    user_func = Profile(after_epoch=True)
    main_loop = setup_mainloop([user_func])

    main_loop.run()
    assert 'profile_training_epoch_after_batch_Profile' in main_loop.log.current_row

def test_exponential_decay():
    lr = shared_floatx(100.0)
    decay = ExponentialDecay(lr,0.01)
    assert_allclose( decay.compute_value(0), 100.0)
    assert_allclose( decay.compute_value(100), 36.787944117144235)

def test_step_decay():
    lr = shared_floatx(100.0)
    decay = StepDecay(lr,[25, 50],.5)
    assert_allclose( decay.compute_value(0), 100.0)
    assert_allclose( decay.compute_value(100), 25.0)

def test_inverse_decay():
    lr = shared_floatx(100.0)
    decay = InverseDecay(lr,.01,.5)
    assert_allclose( decay.compute_value(0), 100.0)
    assert_allclose( decay.compute_value(100), 70.71067811865476)

def test_polynomial_decay():
    lr = shared_floatx(100.0)
    decay = PolynomialDecay(lr,100.0,1.0)
    assert_allclose( decay.compute_value(0), 100.0)
    assert_allclose( decay.compute_value(50), 50.0)
    assert_allclose( decay.compute_value(100), 0.0)
    assert_allclose( decay.compute_value(200), 0.0)

def test_sigmoid_decay():
    lr = shared_floatx(100.0)
    decay = SigmoidDecay(lr,.1,50)
    assert_allclose( decay.compute_value(0), 99.3307113647461)
    assert_allclose( decay.compute_value(50), 50.0)
    assert_allclose( decay.compute_value(100), 0.6692851185798645)

def test_linear_decay():
    lr = shared_floatx(100.0)
    decay = LinearDecay(lr,1.0)
    assert_allclose( decay.compute_value(0.0), 100.0)
    assert_allclose( decay.compute_value(50), 50.0)
    assert_allclose( decay.compute_value(100), 0.0)
    assert_allclose( decay.compute_value(200), 0.0)

def test_compose_decay():
    lr = shared_floatx(100.0)
    decay = ComposeDecay([LinearDecay(lr, 1.0), LinearDecay(lr,2.0), LinearDecay(lr,1.0)],[10,20])
    assert_allclose( decay.compute_value(0), 100.0)
    assert_allclose( decay.compute_value(10), 90.0)
    assert_allclose( decay.compute_value(90), 0.0)

def test_compose_decay_copy():
    lr = shared_floatx(100.0)
    linear1 = LinearDecay(lr, 1.0)
    linear2 = LinearDecay(lr, 1.0)
    decay = ComposeDecay([linear1,linear2],[10])
    assert (decay.variable_modifiers != [linear1,linear2])
    assert(linear2.initial_value == 100.0)
    assert(decay.variable_modifiers[1].initial_value == 90.0)

def test_auc_monitor():
    features = [numpy.array(f, dtype=floatX)
                for f in [[1, 2], [3, 4], [5, 6]]]
    dataset = IterableDataset(dict(features=features))
    datastream = DataStream(dataset)
    test_probs = shared_floatx(numpy.array([
        [0.0, 0.0, 1.0],
        [0.75, 0.25, 0.0],
        [0.0, 0.75, 0.25],
        [0.25, 0.75, 0.0],
    ], dtype=floatX))
    targets = shared_floatx(numpy.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=floatX))
    auc_monitor = AUCMonitor(datastream, test_probs, targets)
    auc_monitor.main_loop = setup_mainloop([])
    auc_monitor.do('after_batch')
    assert_allclose(auc_monitor.main_loop.log[0]['auc'], 0.81944444444444453)

def test_perclass_accuracy_monitor():
    features = [numpy.array(f, dtype=floatX)
                for f in [[1, 2], [3, 4], [5, 6]]]
    dataset = IterableDataset(dict(features=features))
    datastream = DataStream(dataset)
    label_i_to_c = {0:"a", 1:"b", 2:"c"}
    test_probs = shared_floatx(numpy.array([
        [0.0, 0.0, 1.0],
        [0.75, 0.25, 0.0],
        [0.0, 0.75, 0.25],
        [0.25, 0.75, 0.0],
    ], dtype=floatX))
    targets = shared_floatx(numpy.array([
        [2.0],
        [0.0],
        [1.0],
        [2.0]
    ], dtype=floatX))
    perclass_accuracy_monitor = PerClassAccuracyMonitor(datastream,
        prediction=numpy.argmax(test_probs, axis=1),
        targets=targets.ravel(),
        label_i_to_c=label_i_to_c)
    perclass_accuracy_monitor.main_loop = setup_mainloop([])
    perclass_accuracy_monitor.do('after_batch')

    assert perclass_accuracy_monitor.main_loop.log[0]['perclass accuracy_a']==1.0
    assert perclass_accuracy_monitor.main_loop.log[0]['perclass accuracy_b']==1.0
    assert perclass_accuracy_monitor.main_loop.log[0]['perclass accuracy_c']==0.5