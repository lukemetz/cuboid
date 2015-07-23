from cuboid.utils import get_parameters_from_brick
from blocks.bricks import MLP, Rectifier

def test_get_parameters_from_brick():
    mlp = MLP([Rectifier(), Rectifier()], [12, 12, 12])
    mlp.allocate()
    params = get_parameters_from_brick(mlp)
    assert len(params) == 4
