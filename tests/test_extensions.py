import numpy
import theano
import shutil

from theano import tensor
from numpy.testing import assert_allclose

from blocks.datasets import ContainerDataset
from blocks.main_loop import MainLoop
from blocks.algorithms import GradientDescent, Scale
from blocks.utils import shared_floatx
from blocks.model import Model

from blocks.extensions import FinishAfter

from cuboid.extensions import ExperimentSaver, EpochSharedVariableModifier

floatX = theano.config.floatX

def setup_mainloop(extension):
    """Create a MainLoop, register the given extension, supply it with a
        DataStream and a minimal model/cost to optimize.
    """
    features = [numpy.array(f, dtype=floatX)
                for f in [[1, 2], [3, 4], [5, 6]]]
    dataset = ContainerDataset(dict(features=features))

    W = shared_floatx([0, 0], name='W')
    x = tensor.vector('features')
    cost = tensor.sum((x-W)**2)
    cost.name = "cost"

    algorithm = GradientDescent(cost=cost, params=[W],
                                step_rule=Scale(1e-3))


    main_loop = MainLoop(
        model=Model(cost), data_stream=dataset.get_default_stream(),
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


def test_epochsharedvariablemodifier():
    # modified from @bartvm/blocks test_training

    features = [numpy.array(f, dtype=floatX)
                for f in [[1, 2], [3, 4], [5, 6]]]
    dataset = ContainerDataset(dict(features=features))

    W = shared_floatx([0, 0], name='W')
    x = tensor.vector('features')
    cost = tensor.sum((x-W)**2)
    cost.name = "cost"

    step_rule = Scale(0.01)
    sgd = GradientDescent(cost=cost, params=[W],
                          step_rule=step_rule)
    main_loop = MainLoop(
        model=Model(cost), data_stream=dataset.get_default_stream(),
        algorithm=sgd,
        extensions=[
            FinishAfter(after_n_epochs=1),
            EpochSharedVariableModifier(step_rule.learning_rate,
                                   lambda epoch, old: numpy.cast[floatX](old / 10))
            ])

    main_loop.run()

    assert_allclose(step_rule.learning_rate.get_value(), 0.001)

