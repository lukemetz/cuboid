import theano
from theano import tensor as T
import numpy as np
from blocks.algorithms import StepRule

class AdaM(StepRule):
    def __init__(self, alpha=0.0002, b1=0.1, b2=0.001, eps=1e-8):
        self.alpha = alpha
        self.b1 = b1
        self.b2 = b2
        self.eps=eps

    def compute_step(self, p, g):
        """
        reference: http://arxiv.org/abs/1412.6980
        """
        b1 = self.b1
        b2 = self.b2
        eps = self.eps
        alpha = self.alpha

        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        t = theano.shared(np.array(0.0, dtype=theano.config.floatX))

        t_new = t + 1.0
        m_new = b1 * g + (1.0 - b1) * m
        v_new = b2 * g * g + (1 - b2) * v

        m_hat = m_new / (1.0 - T.pow((1.0 - b1), t_new))
        v_hat = v_new / (1.0 - T.pow((1.0 - b2), t_new))

        step = alpha * m_hat / (T.sqrt(v_hat) + eps)
        updates = [(m, m_new), (v, v_new), (t, t_new)]

        return step, updates
