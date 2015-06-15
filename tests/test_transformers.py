from cuboid.transformers import DropSources
from fuel.datasets import IndexableDataset
from numpy.testing import assert_allclose
import numpy as np
from collections import OrderedDict

def test_dropsources():
    stream = IndexableDataset(indexables=OrderedDict([
        ("valid", np.ones((5, 3, 3))),
        ("drop", np.zeros((5, 3, 3))),
        ])).get_example_stream()

    stream = DropSources(stream, ["drop"])

    assert len(stream.sources) == 1
    assert 'valid' in stream.sources

    data = stream.get_epoch_iterator().next()
    assert len(data) == 1
    assert_allclose(data[0], np.ones((3, 3)))
