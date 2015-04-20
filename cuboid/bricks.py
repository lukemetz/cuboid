from blocks.bricks import Brick, Random, Sequence, Feedforward, Initializable, Activation
from blocks.bricks.base import lazy, application
from blocks.bricks import conv
from blocks.initialization import Constant

from blocks.utils import shared_floatx_zeros, pack
from blocks import config

import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from theano.sandbox.cuda.dnn import dnn_conv

import numpy as np


class BatchNormalizationBase(Brick):
    """
    Base class, do not use me.
    Use Either BatchNormalizationConv or BatchNormalization
    """

    seed_rng = np.random.RandomState(config.default_seed)

    @lazy()
    def __init__(self, B_init=Constant(0), Y_init=Constant(1), epsilon=1e-9, seed=None,
            population_mean=None, population_var=None, **kwargs):
        super(BatchNormalizationBase, self).__init__(**kwargs)
        self.eps = epsilon
        self.seed = seed
        self.B_init = B_init
        self.Y_init = Y_init
        self.population_mean = population_mean
        self.population_var = population_var

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
    @lazy()
    def __init__(self, input_dim, **kwargs):
        super(BatchNormalizationConv, self).__init__(**kwargs)
        self.input_dim = input_dim

    def initialize(self):
        self.num_channels = self.input_dim[0]
        super(BatchNormalizationConv, self).initialize()

    def _allocate(self):
        B = shared_floatx_zeros((self.num_channels,), name="B")
        self.params.append(B)

        Y = shared_floatx_zeros((self.num_channels,), name="Y")
        self.params.append(Y)

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        """
        Reference: http://arxiv.org/pdf/1502.03167v2.pdf
        """

        if self.population_mean is None and self.population_var is None:
            minibatch_mean = T.mean(input_, axis=[0, 2, 3]).dimshuffle('x', 0, 'x', 'x')
            minibatch_var = T.mean(T.sqr((input_ - minibatch_mean)), axis=[0, 2, 3]).dimshuffle('x', 0, 'x', 'x')
            norm_x = (input_ - minibatch_mean) / (T.sqrt(minibatch_var + self.eps))
        else:
            theano_mean = T.as_tensor_variable(self.population_mean).dimshuffle('x', 0, 'x', 'x')
            theano_std = T.sqrt(self.population_var + self.eps).dimshuffle('x', 0, 'x', 'x')
            norm_x = (input_ - theano_mean) / theano_std

        B, Y = self.params
        B = B.dimshuffle(('x', 0, 'x', 'x'))
        Y = Y.dimshuffle(('x', 0, 'x', 'x'))
        return norm_x * Y + B

    def get_dim(self, name):
        if name == "input_":
            return self.input_dim
        if name == "output":
            return self.input_dim  
        return super(BatchNormalizationConv, self).get_dim(name)

class BatchNormalization(BatchNormalizationBase):
    @lazy()
    def __init__(self, input_dim, **kwargs):
        super(BatchNormalization, self).__init__(**kwargs)
        self.input_dim = input_dim

    def _allocate(self):
        B = shared_floatx_zeros((self.input_dim,), name="B")
        self.params.append(B)

        Y = shared_floatx_zeros((self.input_dim,), name="Y")
        self.params.append(Y)

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        """
        Reference: http://arxiv.org/pdf/1502.03167v2.pdf
        """
        if self.population_mean is None and self.population_var is None:
            minibatch_mean = T.mean(input_, axis=[0]).dimshuffle('x', 0)
            minibatch_var = T.sqr(input_ - minibatch_mean)
            norm_x = (input_ - minibatch_mean) / (T.sqrt(minibatch_var + self.eps))
        else:
            norm_x = (input_ - self.population_mean) / T.sqrt(self.population_var + self.eps)

        B, Y = self.params
        B = B.dimshuffle(('x', 0))
        Y = Y.dimshuffle(('x', 0))  
        return norm_x * Y + B

    def get_dim(self, name):
        if name == "input_":
            return self.input_dim
        if name == "output":
            return self.input_dim
        return super(BatchNormalizationConv, self).get_dim(name)



srng = RandomStreams(seed=32)

class Dropout(Random):
    @lazy(allocation=['p_drop'])
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
    @lazy(allocation=['input_dim'])
    def __init__(self, input_dim, **kwargs):
        super(Flattener, self).__init__(**kwargs)
        self.input_dim = input_dim

    def get_dim(self, name):
        if name == "input_":
            return self.input_dim  
        if name == "output":
            return np.prod(self.input_dim)

class FilterPool(Brick):
    @lazy()
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


class Convolutional(conv.Convolutional):
    def __init__(self, **kwargs):
        super(Convolutional, self).__init__(**kwargs)

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        if self.use_bias:
            W, b = self.params
        else:
            W, = self.params  

        padding_and_border_mode = self.border_mode
        if self.border_mode == 'same':
            padding_and_border_mode = ((self.filter_size[0] - 1) // 2, (self.filter_size[1] - 1) // 2)

        output = dnn_conv(
            input_, W,
            subsample=self.step,
            border_mode=padding_and_border_mode)
        if self.use_bias:
            output += b.dimshuffle('x', 0, 1, 2)
        return output

    def get_dim(self, name):
        if name == 'input_':
            return (self.num_channels,) + self.image_size
        if name == 'output':
            if self.border_mode == 'same':
                return (self.num_filters,) + self.image_size
            else:
                return ((self.num_filters,) +
                        ConvOp.getOutputShape(self.image_size, self.filter_size,
                                              self.step, self.border_mode))

def flatten(transforms):
    # flatten transforms
    new_trans = []
    for tran in transforms:
        if type(tran) is list:
            new_trans.extend(tran)
        else:
            new_trans.append(tran)
    return new_trans

def init_layer(last_l, last_shape, current_l, index):
    if last_l != None:
        if not (isinstance(last_l, Activation) or isinstance(last_l, Dropout)):
            last_shape = last_l.get_dim("output")

    if hasattr(current_l, 'image_size') and hasattr(current_l, 'num_channels') and len(last_shape) == 3:
        current_l.image_size = last_shape[1:]
        current_l.num_channels = last_shape[0]
    elif hasattr(current_l, 'input_dim'):
        current_l.input_dim = last_shape
    else:
        assert isinstance(current_l, Activation) or isinstance(current_l, Dropout)

    current_l.initialize()
    current_l.name += "_"+str(index)

    return current_l, last_shape, current_l.params

class BrickSequence(Sequence, Initializable, Feedforward):
    """
    Combination class to deal with a sequence of bricks.
    input_dim automatically set on bricks

    Note, this class tries to make its best guess on how inputshapes are propogated. It is not perfect.
    """
    @lazy()
    def __init__(self, input_dim, bricks, **kwargs):
        self.input_dim = input_dim
        self.bricks= flatten(bricks)

        application_methods = [t.apply for t in self.bricks]
        super(BrickSequence, self).__init__(application_methods, **kwargs)

    def _push_allocation_config(self):
        last_l = None
        last_shape = self.input_dim

        for i, l in enumerate(self.bricks):
            last_l, last_shape, p = init_layer(last_l, last_shape, l, i)

    @application
    def apply_inference(self, *args):
        child_input = args
        for brick in self.bricks:
            if isinstance(brick, Dropout):
                continue
            output = brick.apply(*pack(child_input))
            child_input = output
        return output

class LeakyRectifier(Activation):

    def __init__(self, a=0.01, **kwargs):
        self.a = a
        super(LeakyRectifier, self).__init__(**kwargs)

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return T.maximum(input_, self.a*input_)
