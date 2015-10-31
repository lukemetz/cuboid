import numpy as np
from blocks.initialization import NdarrayInitialization
import theano


class ConvIdentity(NdarrayInitialization):
    def __init__(self, scale=1., **kwargs):
        super(ConvIdentity, self).__init__(**kwargs)
        self.scale = scale

    def generate(self, rng, shape):
        w = np.zeros(shape)
        ycenter = shape[2]//2
        xcenter = shape[3]//2
        if shape[0] == shape[1]:
            o_idxs = np.arange(shape[0])
            i_idxs = np.arange(shape[1])
        elif shape[1] < shape[0]:
            o_idxs = np.arange(shape[0])
            tile = np.tile(np.arange(shape[1]), shape[0]/shape[1]+1)
            i_idxs = rng.permutation(tile)[:shape[0]]
        w[o_idxs, i_idxs, ycenter, xcenter] = self.scale
        return w.astype(theano.config.floatX)


class Orthogonal(NdarrayInitialization):
    """ benanne lasagne ortho init (faster than qr approach)"""
    def __init__(self, scale=1.1):
        self.scale = scale

    def generate(self, rng, shape):
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = rng.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        val = self.scale * q[:shape[0], :shape[1]]
        return val.astype(theano.config.floatX)


class UniformIdentity(NdarrayInitialization):
    """Initialize to the identity matrix.
    Only works for 2D arrays. If the number of columns is not equal to the
    number of rows, the array will be truncated or padded with zeros.
    Parameters
    ----------
    mult : float, optional
        Multiply the identity matrix with a scalar. Defaults to 1.
    """
    def __init__(self, min, max):
        self.min, self.max = min, max

    def generate(self, rng, shape):
        if len(shape) != 2:
            raise ValueError
        rows, cols = shape
        mult = rng.uniform(self.min, self.max, size=shape).\
            astype(theano.config.floatX)
        return mult * np.eye(rows, cols, dtype=theano.config.floatX)


class LoadWeights(NdarrayInitialization):
    """ Load weights from a file

    Parameters
    ----------
    path: str
        path to weights npy to load
    """
    def __init__(self, path):
        self.path = path

    def generate(self, rng, shape):
        load = np.load(self.path)
        if load.shape != shape:
            raise ValueError("Weights shapes don't line up. Loaded (%s), "
                             "expected (%s)" % (load.shape, shape))
        return load.astype(theano.config.floatX)
