from blocks.bricks import (Brick, Random, Sequence,\
    Feedforward, Initializable, Activation, FeedforwardSequence, Activation,\
    Linear, Rectifier, Logistic, Activation)
from blocks.bricks import Softmax as BlocksSoftmax
from blocks.bricks.base import lazy, application
from blocks.bricks import conv
from blocks.initialization import Constant, NdarrayInitialization

from blocks.utils import shared_floatx_zeros, pack, shared_floatx_nans
from blocks.config import config
from blocks.roles import add_role, PARAMETER, FILTER, BIAS, WEIGHT
import collections
from theano.sandbox.cuda.basic_ops import gpu_contiguous, gpu_alloc_empty
from theano.sandbox.cuda.dnn import GpuDnnConvDesc, GpuDnnConvGradI

import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from theano.sandbox.cuda.dnn import dnn_conv, GpuDnnConv, dnn_pool

import numpy as np

from blocks.config import config

srng = RandomStreams(seed=32)

class Dropout(Random):
    @lazy(allocation=['p_drop'])
    def __init__(self, p_drop, input_dim=None, **kwargs):
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

    def get_dim(self, name):
        if name in ['input_', 'output'] and self.input_dim != None:
            return self.input_dim
        else:
            return super(Dropout, self).get_dim(name)

class FilterPool(Brick):
    @lazy()
    def __init__(self, fn, input_dim=None, **kwargs):
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


def _rms(x, axis=1):
    return T.sqrt(T.mean(x**2), axis=axis)


class MeanPool(FilterPool):
    def __init__(self, **kwargs):
        super(MeanPool, self).__init__(fn=T.mean, **kwargs)


class RMSPool(FilterPool):
    def __init__(self, **kwargs):
        super(MeanPool, self).__init__(fn=_rms, **kwargs)


class Convolutional(Initializable):
    @lazy(allocation=['input_dim', 'num_filters', 'filter_size'])
    def __init__(self, input_dim, num_filters, filter_size, pad=(1, 1),
                 stride=(1, 1), **kwargs):
        super(Convolutional, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.pad = pad
        self.stride = stride

    def _allocate(self):
        if not isinstance(self.input_dim, collections.Iterable)\
           or len(self.input_dim) != 3:

            raise ValueError("`input_dim` on Convolutional(%s) "
                             "should be iterable and have a inputdim of 3. "
                             "Got %s"
                             %(self.name, str(self.input_dim)))
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
            subsample=self.stride,
            border_mode=self.pad)
        if self.use_bias:
            output += b.dimshuffle('x', 0, 'x', 'x')
        return output

    def get_dim(self, name):
        if name == "output":
            i1_type = type(self.input_dim[1])
            i2_type = type(self.input_dim[2])
            if  i1_type != str and i2_type != str:
                ishape = (self.input_dim[0], 'x', self.input_dim[1], self.input_dim[2])
                kshape = (self.num_filters, 'x', self.filter_size[0], self.filter_size[1])
                border_mode = self.pad
                subsample = self.stride
                oshape = GpuDnnConv.get_out_shape(ishape, kshape, border_mode, subsample)
                return (oshape[1], oshape[2], oshape[3])
            else:
                # TODO manage the case where either input_dim[{1, 2}] is not a str
                return (self.num_filters, self.input_dim[1], self.input_dim[2])
        else:
            return super(Conv1D, self).get_dim(name)


class Pooling(Brick):
    @lazy(initialization=['input_dim'])
    def __init__(self, input_dim, pooling_size=(2,2), stride=None, pad=(0,0), **kwargs):
        super(Pooling, self).__init__(**kwargs)
        self.pad = pad
        self.input_dim = input_dim
        self.stride = stride
        self.pooling_size=pooling_size

    def apply(self, input_):
        raise NotImplemented

    def get_dim(self, name):
        if name == 'output':
            c, x, y = self.input_dim
            if type(x) == str and type(y) == str:
                return (c, x, y)
            px, py = self.pad
            if self.stride:
                sx, sy = self.stride
            else:
                sx, sy = self.pooling_size

            kx, ky = self.pooling_size
            return (c, (x + 2 * px - kx) // sx + 1,\
                       (y + 2 * py - ky) // sy + 1)

class MaxPooling(Pooling):
    @application(inputs=["input_"], outputs=["output"])
    def apply(self, input_):
        stride = self.stride
        if not stride:
            stride = self.pooling_size
        return dnn_pool(input_, ws=self.pooling_size, stride=stride, pad=self.pad)


class MeanPooling(Pooling):
    @application(inputs=["input_"], outputs=["output"])
    def apply(self, input_):
        stride = self.stride
        if not stride:
            stride = self.pooling_size
        return dnn_pool(input_, ws=self.pooling_size, stride=stride,
                        pad=self.pad, mode='average_inc_pad')


class RMSPooling(Pooling):
    @application(inputs=["input_"], outputs=["output"])
    def apply(self, input_):
        stride = self.stride
        if not stride:
            stride = self.pooling_size
        inter = input_**2
        inter = dnn_pool(inter, ws=self.pooling_size, stride=stride,
                         pad=self.pad, mode='average_inc_pad')
        return T.sqrt(inter)


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
    @lazy(allocation=["input_dim"])
    def __init__(self, input_dim, lists, **kwargs):
        application_methods = []
        for entry in lists:
            if hasattr(entry, '__call__'):
                application_methods.append(entry)
            else:
                application_methods.append(entry.apply)
        super(DefaultsSequence, self).__init__(application_methods=application_methods, **kwargs)

        self.input_dim = input_dim
        self.output_dim = None

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


    def _push_allocation_config(self):
        input_dim = self.input_dim
        for brick, application_method in zip(self.children, self.application_methods):
            print input_dim, brick.name
            # hack as Activations don't have get_dim
            if not isinstance(brick, Activation) and not isinstance(brick, BlocksSoftmax)\
               and not isinstance(brick, Softmax):

                brick.input_dim = input_dim
                brick.push_allocation_config()
                input_dim = brick.get_dim(brick.apply.outputs[0])

        self.output_dim = input_dim

        self.names = {}
        for child in self.children:
            if child.name in self.names:
                self.names[child.name] += 1
                child.name += "_" + str(self.names[child.name])
            else:
                self.names[child.name] = 0

    def get_dim(self, name):
        if name == "output":
            return self.output_dim
        else:
            return super(DefaultsSequence, self).get_dim(name)

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
                # TODO fix this awful hack
                if hasattr(self.input_dim, "__iter__") and\
                        not all([type(t) != str for t in self.input_dim]):
                    raise AttributeError("Cannot pass in variable lengths"
                        "to Flattener. The input shape gotten is %s" % str(self.input_dim))

                return np.prod(self.input_dim)
            else:
                raise ValueError("No input_dim set on Flattener")
        else:
            return super(Flattener, self).get_dim(name)

class Highway(Initializable, Feedforward):
    """ Implements highway networks outlined in [1]

    y = H(x,WH)T(x,WT) + x(1-T(x,WT))

    Highway networks have the same input dimension and output dimension

    Parameters
    ----------
    input_dim: int
        number of input/output dimensions for the network
    output_activation: Activation
        activation function applied to x and the hidden weights
    transform_activation: Activation
        activation function applied to x and the transform weights
    [1] http://arxiv.org/pdf/1505.00387v1.pdf
    """

    @lazy(allocation=['input_dim'])
    def __init__(self, input_dim, output_activation=None, transform_activation=None, **kwargs):
        super(Highway, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = input_dim

        if output_activation == None:
            output_activation = Rectifier()

        if transform_activation == None:
            transform_activation = Logistic()

        self._linear_h = Linear(name="linear_h", input_dim=input_dim, output_dim=input_dim)
        self._linear_t = Linear(name="linear_t", input_dim=input_dim, output_dim=input_dim)
        self._output_activation = output_activation
        self._transform_activation = transform_activation
        self.children = [self._linear_h, self._linear_t, self._output_activation, self._transform_activation]

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        h = self._output_activation.apply(self._linear_h.apply(input_))
        t = self._transform_activation.apply(self._linear_t.apply(input_))

        return h*t+input_*(1-t)


class SteeperSigmoid(Activation):
    def __init__(self, scale=3.75, **kwargs):
        super(SteeperSigmoid, self).__init__(**kwargs)
        self.scale = scale

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return 1./(1. + T.exp(-self.scale * input_))


class PReLU(Initializable):
    """Rectifier with a learned negative slope.

    Modified from https://github.com/mila-udem/blocks-extras/pull/18
    Thanks janchorowski!

    Parameters
    ----------
    dim : int
        The dimension of the input. Required if slopes are not tied
        by :meth:`~.Brick.allocate`.
    tie_slopes : bool, default False
        Do all units use the same negative slope, or is the slope specific
        to each unit? When slopes are tied it is not necessary to specify
        the dim - it is not used.
    slopes_init : object, default Constant(0.25)
        A `NdarrayInitialization` instance which will be used by to
        initialize the slopes matrix. Required by
        :meth:`~.Brick.initialize`.

    See also: [He2015]_.

    .. [He2015] K. He at al. Delving Deep into Rectifiers:
        Surpassing Human-Level Performance on ImageNet Classification

    """
    @lazy(allocation=['input_dim'])
    def __init__(self, input_dim, tie_slopes=False, slopes_init=None, **kwargs):
        super(PReLU, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.tie_slopes = tie_slopes
        if slopes_init is None:
            slopes_init = Constant(0.25)
        self.slopes_init = slopes_init

    def _allocate(self):
        dim = None
        if isinstance(self.input_dim, collections.Iterable):
            if len(self.input_dim) == 3:  # CNN
                dim = self.input_dim[0]
            else:
                raise NotImplemented
        else:
            dim = self.input_dim

        if self.tie_slopes:
            dim = 1
        a = shared_floatx_nans((dim,), name='a')
        add_role(a, PARAMETER)
        self.parameters.append(a)

    def _initialize(self):
        a, = self.parameters
        self.slopes_init.initialize(a, self.rng)

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        """Apply the PReLU activation: x if x>=0 else a*x."""
        a, = self.parameters
        if self.tie_slopes:
            a = T.addbroadcast(a, 0)
        else:
            a = a.dimshuffle('x', 0, 'x', 'x')
        return T.switch(input_ >= 0, input_, input_ * a)

    def get_dim(self, name):
        if name in ['input_', 'output']:
            return self.input_dim
        super(PReLU, self).get_dim(name)


class Softmax(Brick):
    """ Softmax that can or cannot use built in theano softmax
    """
    def __init__(self, built_in=False, **kwargs):
        super(Softmax, self).__init__(**kwargs)
        self.built_in = built_in

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        if self.built_in:
            return T.nnet.softmax(input_)
        else:
            e_x = T.exp(input_ - input_.max(axis=1).dimshuffle(0, 'x'))
            return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')


class Deconvolutional(Convolutional):
    @lazy(allocation=['input_dim', 'num_filters', 'filter_size'])
    def __init__(self, input_dim, num_filters, filter_size, pad=(1, 1),
                 stride=(1, 1), **kwargs):
        super(Deconvolutional, self).__init__(input_dim,
                                              num_filters,
                                              filter_size,
                                              pad,
                                              stride,
                                              **kwargs)

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        if self.use_bias:
            W, b = self.parameters
        else:
            W, = self.parameters
        W = W.dimshuffle(1, 0, 2, 3)
        img = gpu_contiguous(input_)
        kerns = gpu_contiguous(W)
        desc = GpuDnnConvDesc(border_mode=self.pad, subsample=self.stride,
                              conv_mode='conv')(gpu_alloc_empty(img.shape[0],
                                                                kerns.shape[1],
                                                                img.shape[2]*self.stride[0],
                                                                img.shape[3]*self.stride[1]).shape,
                                                kerns.shape)
        out = gpu_alloc_empty(img.shape[0], kerns.shape[1],
                              img.shape[2]*self.stride[0],
                              img.shape[3]*self.stride[1])
        output = GpuDnnConvGradI()(kerns, img, out, desc)
        if self.use_bias:
            output += b.dimshuffle('x', 0, 'x', 'x')
        return output
