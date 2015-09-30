from cuboid.evaluators import DataStreamEvaluator
from cuboid.transformers import DropSources
from fuel.datasets import IndexableDataset
from numpy.testing import assert_allclose
import numpy as np
from collections import OrderedDict
import theano.tensor as T

def test_datastream_evaluator():
    stream = IndexableDataset(indexables=OrderedDict([
        ("data", np.ones((10, 4, 9), dtype="float32")),
        ])).get_example_stream()

    x = T.matrix("data")
    mon = x.sum(axis=1)
    mon.name = "mon"

    evaluator = DataStreamEvaluator([mon])
    results = evaluator.evaluate(stream)
    assert set(results.keys()) == set(['mon'])

    assert_allclose(results['mon'], np.ones((4*10))*9)
