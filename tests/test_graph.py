from cuboid.graph import get_bricks_matching
from blocks.bricks import Linear, Rectifier
from cuboid.bricks import DefaultsSequence, Dropout
from blocks.model import Model
import theano.tensor as T

def test_get_bricks_matching():
    seq = DefaultsSequence(input_dim=9, lists=[
        Linear(output_dim=10),
        Dropout(p_drop=0.5),
        Rectifier(),
        Linear(output_dim=12, name="lin2"),
        Dropout(p_drop=0.5),
        Rectifier()
    ])

    x = T.matrix('input')
    y = seq.apply(x)

    bricks = get_bricks_matching(Model([y]), lambda x: type(x) == Dropout)
    assert len(bricks) == 2
    assert type(bricks[0]) == Dropout

    bricks = get_bricks_matching(seq, lambda x: type(x) == Dropout)
    assert len(bricks) == 2
    assert type(bricks[0]) == Dropout
