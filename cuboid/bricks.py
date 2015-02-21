from blocks.bricks import Brick, Random
from blocks.bricks.base import lazy, application
from blocks.bricks import conv

from blocks.utils import shared_floatx_zeros
from blocks.utils import shared_floatx_zeros
from blocks import config

import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import numpy as np


class BatchNormalizationBase(Brick):
    """
    Base class, do not use me.
    Use Either BatchNormalizationConv or BatchNormalization
    """

    seed_rng = np.random.RandomState(config.default_seed)

    @lazy
    def __init__(self, B_init, Y_init, epsilon=1e-9, seed=None, **kwargs):
        super(BatchNormalizationBase, self).__init__(**kwargs)
        self.eps = epsilon
        self.seed = seed
        self.B_init = B_init
        self.Y_init = Y_init

    @property
    def seed(self):
        if getattr(self, '_seed', None) is not None:
            return self._seed
        else:
            self._seed = self.seed_rng.randint(np.iinfo(np.int32).max)
            return self._seed

    @seed.setter
    def seed(self, value):
        if hasattr(self, '_seed'):
            raise AttributeError("seed already set")
        self._seed = value

    @property
    def rng(self):
        if getattr(self, '_rng', None) is not None:
                return self._rng
        else:
            return np.random.RandomState(self.seed)

    @rng.setter
    def rng(self, rng):
        self._rng = rng

    def _initialize(self):
        B, Y = self.params
        self.B_init.initialize(B, self.rng)
        self.Y_init.initialize(Y, self.rng)


class BatchNormalizationConv(BatchNormalizationBase):
    @lazy
    def __init__(self, input_shape, **kwargs):
        super(BatchNormalizationConv, self).__init__(**kwargs)
        self.input_shape = input_shape
        self.num_channels = input_shape[0]

    def _allocate(self):
        B = shared_floatx_zeros((self.num_channels,))
        self.params.append(B)

        Y = shared_floatx_zeros((self.num_channels,))
        self.params.append(Y)

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        """
        Reference: http://arxiv.org/pdf/1502.03167v2.pdf
        """
        minibatch_mean = T.mean(input_, axis=[0, 2, 3]).dimshuffle('x', 0, 'x', 'x')
        minibatch_var = T.var(input_, axis=[0, 2, 3]).dimshuffle('x', 0, 'x', 'x')
        norm_x = (input_ - minibatch_mean) / (T.sqrt(minibatch_var + self.eps))
        B, Y = self.params
        B = B.dimshuffle(('x', 0, 'x', 'x'))
        Y = Y.dimshuffle(('x', 0, 'x', 'x'))
        return norm_x * Y + B

    def get_dim(self, name):
        if name == "input_":
            return self.input_shape
        if name == "output":
            return self.input_shape
        super(BatchNormalizationConv, self).get_dim(name)

class BatchNormalization(BatchNormalizationBase):
    @lazy
    def __init__(self, input_dim, **kwargs):
        super(BatchNormalization, self).__init__(**kwargs)
        self.input_dim = input_dim

    def _allocate(self):
        B = shared_floatx_zeros((self.input_dim,))
        self.params.append(B)

        Y = shared_floatx_zeros((self.input_dim,))
        self.params.append(Y)

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        """
        Reference: http://arxiv.org/pdf/1502.03167v2.pdf
        """
        minibatch_mean = T.mean(input_, axis=[0]).dimshuffle('x', 0)
        minibatch_var = T.var(input_, axis=[0]).dimshuffle('x', 0)
        norm_x = (input_ - minibatch_mean) / (T.sqrt(minibatch_var + self.eps))
        B, Y = self.params
        B = B.dimshuffle(('x', 0))
        Y = Y.dimshuffle(('x', 0))
        return norm_x * Y + B

    def get_dim(self, name):
        if name == "input_":
            return self.input_dim
        if name == "output":
            return self.input_dim
        super(BatchNormalizationConv, self).get_dim(name)



srng = RandomStreams(seed=32)

class Dropout(Random):
    @lazy
    def __init__(self, p_drop, **kwargs):
        super(Dropout, self).__init__(**kwargs)
        self.p_drop = p_drop

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        if self.p_drop != 0:
            retain_prob = 1 - self.p_drop
            mask = srng.binomial(input_.shape, p=retain_prob, dtype='int32').astype('float32')
            return input_ / retain_prob * mask
        else:
            return input_

class Flattener(conv.Flattener):
    @lazy
    def __init__(self, input_dim, **kwargs):
        super(Flattener, self).__init__(**kwargs)
        self.input_dim = input_dim

    def get_dim(self, name):
        if name == "input_":
            return self.input_dim
        if name == "output":
            return np.prod(self.input_dim)

class FilterPool(Brick):
    @lazy
    def __init__(self, fn, input_dim, **kwargs):
        super(FilterPool, self).__init__(**kwargs)
        self.fn = fn
        self.input_dim = input_dim

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return self.fn(input_.reshape((input_.shape[0], input_.shape[1], -1)), axis=2)

    def get_dim(self, name):
        if name == "input_":
            return self.input_dim
        elif name == "output":
            return self.input_dim[0]

