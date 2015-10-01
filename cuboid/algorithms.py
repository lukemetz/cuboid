import theano
import numpy as np
from blocks.algorithms import StepRule


class NAG(StepRule):
    def __init__(self, lr=0.01, momentum=0.9, **kwargs):
        self.lr = theano.shared(np.array(lr, dtype=theano.config.floatX))
        self.momentum = theano.shared(np.array(momentum,
                                               dtype=theano.config.floatX))

    def compute_step(self, p, g):
        m = theano.shared(p.get_value() * 0.)
        v = (self.momentum * m) - (self.lr * g)
        updates = [(m, v)]

        step = -self.momentum * v + self.lr * g
        return step, updates
