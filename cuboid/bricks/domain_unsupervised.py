import numpy as np
import theano
from theano.scalar import UnaryScalarOp, same_out
from theano.gof import local_optimizer, Op, Apply
from theano.sandbox.cuda import register_opt, GpuOp
import theano.tensor as T

from blocks.bricks.base import Brick, application

class GradientReversalOp(GpuOp):
    """
    References
    ----------
    .. http://arxiv.org/abs/1409.7495
    """
    def make_node(self, x):
        return Apply(self, [x], [x.type()])

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        return -gz,

gradient_reversal = GradientReversalOp()

@register_opt()
@local_optimizer([GradientReversalOp])
def local_remove_gradient_reversal(node):
    """gradient_reversal(x)
       -> x
    """
    if (isinstance(node.op, GradientReversalOp)):
        return (node.inputs[0], )

class GradientReversal(Brick):
    """
    Identity on the forward pass, but has the negative of gradient
    As described in [1].

    References
    ----------
    ..[1] http://arxiv.org/abs/1409.7495
    """

    @application(inputs=["input_"], outputs=["output"])
    def apply(self, input_):
        return gradient_reversal(input_)

class BatchwiseSplit(Brick):
    """
    Split the batch into two pieces at split_idx.
    Used for 2 headed models that need the same features
    """
    def __init__(self, split_idx, **kwargs):
        self.split_idx = split_idx
        super(BatchwiseSplit, self).__init__(**kwargs)

    @application(inputs=["input_"], outputs=["first_half", "second_half"])
    def apply(self, input_):
        return input_[0:self.split_idx], input_[self.split_idx:]

from blocks.bricks.cost import CategoricalCrossEntropy

class DomainUnsupervisedCost(Brick):
    """
    Domain unsupervised head as described [1].

    Parameters
    ---------
    split_idx: int
        location in the batch denoting where unsupervised data starts.

    domain_classifier: ~`blocks.bricks.Brick`
        brick containing transform to binary classifier
    References
    ----------
    .. [1] http://arxiv.org/abs/1409.7495
    """
    def __init__(self, split_idx, domain_classifier, **kwargs):
        super(DomainUnsupervisedCost, self).__init__(**kwargs)
        batchwise_split = BatchwiseSplit(split_idx)
        gradient_reversal = GradientReversal()
        cost = CategoricalCrossEntropy()

        self.children = [gradient_reversal, domain_classifier, batchwise_split, cost]

    @application(inputs=['input_'], outputs=['cost'])
    def apply(self, input_):
        gradient_reversal, domain_classifier, batchwise_split, cost = self.children
        reversed_ = gradient_reversal.apply(input_)
        targets = domain_classifier.apply(reversed_)
        supervised, unsupervised = batchwise_split.apply(targets)
        ones = T.ones((supervised.shape[0],), dtype='int32')
        zeros = T.zeros((unsupervised.shape[0],), dtype='int32')
        return cost.apply(ones, supervised) + cost.apply(zeros, unsupervised)
