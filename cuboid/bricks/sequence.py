from theano.tensor.signal.downsample import max_pool_2d

from theano.sandbox.cuda.dnn import dnn_conv
from theano.sandbox.cuda.dnn import dnn_pool

from theano import tensor

from blocks.bricks import Initializable, Feedforward
from blocks.bricks.base import Brick, lazy, application
from blocks.roles import add_role, FILTER, BIAS
from blocks.utils import shared_floatx_nans

class Conv1D(Initializable):
    """Perform a 1D convolution

    Parameters
    ----------
    filter_length : int
    num_filters : int
    input_dim : int
    step : int
    pad : int
    """

    @lazy(allocation=['filter_length', 'num_filters', 'input_dim'])
    def __init__(self, filter_length, num_filters, input_dim, step=1, pad=1, **kwargs):
        super(Conv1D, self).__init__(**kwargs)

        self.filter_length = filter_length
        self.num_filters = num_filters
        self.input_dim = input_dim
        self.step = step
        self.pad = pad

    def _allocate(self):
        W = shared_floatx_nans((self.num_filters, self.input_dim,
            self.filter_length, 1), name='W')
        add_role(W, FILTER)
        self.parameters.append(W)

        if self.use_bias:
            b = shared_floatx_nans((self.num_filters, ), name='b')
            add_role(b, BIAS)
            self.parameters.append(b)

    def _initialize(self):
        W, b = self.parameters
        self.biases_init.initialize(b, self.rng)
        self.weights_init.initialize(W, self.rng)

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        """Perform the convolution.

        Parameters
        ----------
        input_ : :class:`~tensor.TensorVariable`
            A 3D tensor with axes batch size, sequence, features

        Returns
        -------
        output : :class: `~tensor.TensorVariable`
            A 3D tensor of filtered sequences with axes batch size, sequence,
            filter map response
        """
        W, b = self.parameters
        shuffled = input_.dimshuffle(0, 2, 1, 'x')

        output = dnn_conv(
                shuffled, W,
                subsample=(self.step, 1),
                border_mode=(self.pad, 0))

        sequence_out = output[:, :, :, 0].dimshuffle(0, 2, 1)
        return sequence_out + b.dimshuffle('x', 0)

class MaxPooling1D(Initializable, Feedforward):
    """Max pooling for 1D sequences

    Parameters
    ----------
    pooling_length : int
        The downsample factor
    step : int, optional
        shifts between pooling region. By default this is equal to `pooling_length`.
    """

    @lazy(allocation=['pooling_length'])
    def __init__(self, pooling_length, step=None, **kwargs):
        super(MaxPooling1D, self).__init__(**kwargs)

        self.pooling_length = pooling_length
        self.step = step

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        """Apply the pooling (subsampling) transform

        Parameters
        ----------
        input_ : :class:`~tensor.TensorVariable`
            3D tensor with axes batch size, sequence, features

        Returns
        -------
        output: :class:`~tensor.TensorVariable`
            3D tensor with axes batch size, sequence, features
        """
        shuffled = input_.dimshuffle(0, 2, 1, 'x')

        if self.step == None:
            st = (self.pooling_length, 1)
        else:
            st = (self.step, 1)
        output = dnn_pool(shuffled, (self.pooling_length, 1), stride=st)

        sequence_out = output[:, :, :, 0].dimshuffle(0, 2, 1)

        return sequence_out
