from blocks.bricks import (Brick, Random, Sequence,\
    Feedforward, Initializable, Activation, FeedforwardSequence, Activation)
from blocks.bricks.base import lazy, application
from blocks.bricks import conv
from blocks.initialization import Constant, NdarrayInitialization

from blocks.utils import shared_floatx_zeros, pack, shared_floatx_nans
from blocks.config import config
from blocks.roles import add_role, PARAMETER, FILTER, BIAS, WEIGHT


import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from theano.sandbox.cuda.dnn import dnn_conv, GpuDnnConv

import numpy as np

from blocks.config import config


class BatchNormalization(Brick):
    seed_rng = np.random.RandomState(config.default_seed)

    @lazy(allocation=['input_dim'])
    def __init__(self, input_dim, epsilon=1e-8, **kwargs):
        super(BatchNormalization, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.bn_active = True
        self.e = epsilon

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

    @property
    def naxes(self):
        if isinstance(self.input_dim, int):
            return 2
        else:
            return len(self.input_dim) + 1

    def _allocate(self):
        naxes = self.naxes
        if naxes == 2:
            dim = self.input_dim
        elif naxes == 4:
            dim = self.input_dim[0]
        elif naxes == 3:
            dim = self.input_dim[-1]
        else:
            raise NotImplementedError
        self.g = shared_floatx_nans((dim, ), name='g')
        self.b = shared_floatx_nans((dim, ), name='b')
        self.u = shared_floatx_nans((dim, ), name='u')
        self.s = shared_floatx_nans((dim, ), name='s')
        self.n = shared_floatx_nans((1,), name='n')
        self.parameters = [self.g, self.b]
        add_role(self.g, WEIGHT)
        add_role(self.b, BIAS)

    def _initialize(self):
        Constant(1).initialize(self.g, self.rng)
        Constant(0).initialize(self.b, self.rng)
        Constant(0).initialize(self.u, self.rng)
        Constant(0).initialize(self.s, self.rng)
        Constant(0).initialize(self.n, self.rng)

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        X = input_
        naxes = self.naxes
        if naxes == 4: #CNN
            if self.bn_active:
                u = T.mean(X, axis=[0, 2, 3])
            else:
                u = self.u/self.n
            b_u = u.dimshuffle('x', 0, 'x', 'x')
            if self.bn_active:
                s = T.mean(T.sqr(X - b_u), axis=[0, 2, 3])
            else:
                s = self.s/self.n
            X = (X - b_u) / T.sqrt(s.dimshuffle('x', 0, 'x', 'x') + self.e)
            X = self.g.dimshuffle('x', 0, 'x', 'x')*X + self.b.dimshuffle('x', 0, 'x', 'x')
        elif naxes == 3: #RNN
            if self.bn_active:
                u = T.mean(X, axis=[0, 1])
            else:
                u = self.u/self.n
            b_u = u.dimshuffle('x', 'x', 0)
            if self.bn_active:
                s = T.mean(T.sqr(X - b_u), axis=[0, 1])
            else:
                s = self.s/self.n
            X = (X - b_u) / T.sqrt(s.dimshuffle('x', 'x', 0) + self.e)
            X = self.g.dimshuffle('x', 'x', 0)*X + self.b.dimshuffle('x', 'x', 0)
        elif naxes == 2: #FC
            if self.bn_active:
                u = T.mean(X, axis=0)
            else:
                u = self.u/self.n
            if self.bn_active:
                s = T.mean(T.sqr(X - u), axis=0)
            else:
                s = self.s/self.n
            X = (X - u) / T.sqrt(s + self.e)
            X = self.g*X + self.b
        else:
            raise NotImplementedError
        return X

    def get_dim(self, name):
        if name == "input_" or name == "output":
            return self.input_dim
        else:
            return super(BatchNormalization, self).get_dim(name)

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


class Convolutional_Old(conv.Convolutional):
    def __init__(self, **kwargs):
        super(Convolutional, self).__init__(**kwargs)

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        if self.use_bias:
            W, b = self.parameters
        else:
            W, = self.parameters

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

class Convolutional(Initializable):
    @lazy(allocation=['input_dim', 'num_filters', 'filter_size'])
    def __init__(self, input_dim, num_filters, filter_size, pad=(1,1),
            step=(1,1), **kwargs):
        super(Convolutional, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.pad = pad
        self.step = step

    def _allocate(self):
        num_channels = self.input_dim[0]
        W = shared_floatx_nans((self.num_filters, num_channels) +
                               self.filter_size, name='W')
        add_role(W, FILTER)
        self.parameters.append(W)
        self.add_auxiliary_variable(W.norm(2), name='W_norm')
        if self.use_bias:
            b = shared_floatx_nans((self.num_filters,), name='b')
            add_role(b, BIAS)

            self.parameters.append(b)
            self.add_auxiliary_variable(b.norm(2), name='b_norm')

    def _initialize(self):
        if self.use_bias:
            W, b = self.parameters
            self.biases_init.initialize(b, self.rng)
        else:
            W, = self.parameters
        self.weights_init.initialize(W, self.rng)

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        if self.use_bias:
            W, b = self.parameters
        else:
            W, = self.parameters
        output = dnn_conv(
            input_, W,
            subsample=self.step,
            border_mode=self.pad)
        print b.ndim, output.ndim
        if self.use_bias:
            output += b.dimshuffle('x', 0, 'x', 'x')
        return output

    def get_dim(self, name):
        if name == "output":
            ishape = (self.input_dim[0], 'x', self.input_dim[1], self.input_dim[2])
            kshape = (self.num_filters, 'x', self.filter_size[0], self.filter_size[1])
            border_mode = self.pad
            subsample = self.step
            oshape = GpuDnnConv.get_out_shape(ishape, kshape, border_mode, subsample)
            return (oshape[1], oshape[2], oshape[3])
        else:
            return super(Conv1D, self).get_dim(name)


class LeakyRectifier(Activation):
    def __init__(self, a=0.01, **kwargs):
        self.a = a
        super(LeakyRectifier, self).__init__(**kwargs)

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return T.maximum(input_, self.a*input_)

class FuncBrick(Brick):
    """
    Brick for use with inline functions.
    Useful for a quick wrap around theano functions.

    Parameters
    ----------
    func: callable
        function to be called
    """
    def __init__(self, func, shape_func=None, input_dim=None, **kwargs):
        self.func = func
        self.input_dim = input_dim
        self.shape_func = shape_func
        super(FuncBrick, self).__init__(**kwargs)

    @application(outputs=['output'])
    def apply(self, *args):
        return self.func(*args)

    def get_dim(self, name):
        if self.shape_func and self.input_dim:
            if name == "output":
                return self.shape_func(self.input_dim)
        else:
            raise ValueError("shape_func not set")
        return super(FuncBrick, self).get_dim(name)

class DefaultsSequence(FeedforwardSequence, Initializable):
    def __init__(self, input_dim, lists, **kwargs):
        application_methods = []
        for entry in lists:
            if hasattr(entry, '__call__'):
                application_methods.append(entry)
            else:
                application_methods.append(entry.apply)
        super(DefaultsSequence, self).__init__(application_methods=application_methods, **kwargs)

        self.input_dim = input_dim

    def _push_initialization_config(self):
        # don't overwrite if parameters are set
        for child in self.children:
            if isinstance(child, Initializable):
                child.rng = self.rng
                if (self.weights_init and
                        not isinstance(child.weights_init, NdarrayInitialization)):
                    child.weights_init = self.weights_init
        if hasattr(self, 'biases_init') and self.biases_init:
            for child in self.children:
                if ((isinstance(child, Initializable) and
                        hasattr(child, 'biases_init')) and
                        not isinstance(child.biases_init, NdarrayInitialization)):
                    child.biases_init = self.biases_init

        self.names = {}
        for child in self.children:
            if child.name in self.names:
                self.names[child.name] += 1
                child.name += "_" + str(self.names[child.name])
            else:
                self.names[child.name] = 0

    def _push_allocation_config(self):
        input_dim = self.input_dim
        for brick, application_method in zip(self.children, self.application_methods):
            print input_dim, brick.name
            # hack as Activations don't have get_dim
            if not isinstance(brick, Activation):
                brick.input_dim = input_dim
                input_dim = brick.get_dim(application_method.outputs[0])

class Flattener(Brick):
    """Flattens the input.
    It may be used to pass multidimensional objects like images or feature
    maps of convolutional bricks into bricks which allow only two
    dimensional input (batch, features) like MLP.
    """
    @lazy(allocation=['input_dim'])
    def __init__(self, input_dim, **kwargs):
        super(Flattener, self).__init__(**kwargs)
        self.input_dim = input_dim

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return input_.flatten(ndim=2)

    def get_dim(self, name):
        if name == "output":
            if self.input_dim:
                return np.prod(self.input_dim)
            else:
                raise ValueError("No input_dim set on Flattener")
        else:
            return super(Flattener, self).get_dim(name)
