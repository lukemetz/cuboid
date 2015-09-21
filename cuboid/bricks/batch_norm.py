from blocks.main_loop import MainLoop
from blocks.algorithms import TrainingAlgorithm
from blocks.roles import add_role, AuxiliaryRole, ParameterRole
from blocks.graph import ComputationGraph
import numpy as np

from blocks.bricks.base import Brick, lazy, application
from blocks.config import config
from blocks.utils import shared_floatx_nans
from blocks.roles import WEIGHT, BIAS
from blocks.initialization import Constant
from theano import tensor as T
from blocks.filter import VariableFilter, get_brick
from blocks.utils import dict_union
import theano
import logging
from collections import OrderedDict
from cuboid.graph import get_parameter_name
from blocks.extensions import FinishAfter, ProgressBar

logger = logging.getLogger(__name__)


class BatchNormPopulationRole(AuxiliaryRole):
    pass

#: Variable for batchnorm populations
BATCHNORM_POPULATION = BatchNormPopulationRole()

class BatchNormalization(Brick):
    seed_rng = np.random.RandomState(config.default_seed)

    @lazy(allocation=['input_dim'])
    def __init__(self, input_dim, epsilon=1e-8, use_population=False, accumulate=False, **kwargs):
        super(BatchNormalization, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.use_population = use_population
        self.e = epsilon
        self.accumulate = accumulate
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
        add_role(self.g, WEIGHT)
        add_role(self.b, BIAS)
        self.parameters = [self.g, self.b]

        # parameters for inference
        self.u = shared_floatx_nans((dim, ), name='u')
        self.s = shared_floatx_nans((dim, ), name='s')
        self.n = shared_floatx_nans((1,), name='n')

        self.add_auxiliary_variable(self.u, roles=[BATCHNORM_POPULATION])
        self.add_auxiliary_variable(self.s, roles=[BATCHNORM_POPULATION])
        self.add_auxiliary_variable(self.n, roles=[BATCHNORM_POPULATION])

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
            if self.use_population:
                u = self.u/self.n
            else:
                u = T.mean(X, axis=[0, 2, 3])
            b_u = u.dimshuffle('x', 0, 'x', 'x')
            if self.use_population:
                s = self.s/self.n
            else:
                s = T.mean(T.sqr(X - b_u), axis=[0, 2, 3])
            X = (X - b_u) / T.sqrt(s.dimshuffle('x', 0, 'x', 'x') + self.e)
            X = self.g.dimshuffle('x', 0, 'x', 'x')*X + self.b.dimshuffle('x', 0, 'x', 'x')
        elif naxes == 3: #RNN
            if self.use_population:
                u = self.u/self.n
            else:
                u = T.mean(X, axis=[0, 1])
            b_u = u.dimshuffle('x', 'x', 0)
            if self.use_population:
                s = self.s/self.n
            else:
                s = T.mean(T.sqr(X - b_u), axis=[0, 1])
            X = (X - b_u) / T.sqrt(s.dimshuffle('x', 'x', 0) + self.e)
            X = self.g.dimshuffle('x', 'x', 0)*X + self.b.dimshuffle('x', 'x', 0)
        elif naxes == 2: #FC
            if self.use_population:
                u = self.u/self.n
            else:
                u = T.mean(X, axis=0)
            if self.use_population:
                s = self.s/self.n
            else:
                s = T.mean(T.sqr(X - u), axis=0)
            X = (X - u) / T.sqrt(s + self.e)
            X = self.g*X + self.b
        else:
            raise NotImplementedError

        if self.accumulate:
            if self.use_population == True:
                raise Exception("use_population is set to true as well as with accumulation. ",
                    "This is not possible as there is nothing to take the population of.")

            self.updates[self.u] = self.u + u
            self.updates[self.s] = self.s + s
            self.updates[self.n] = self.n + 1

        return X

    def get_dim(self, name):
        if name == "input_" or name == "output":
            return self.input_dim
        else:
            return super(BatchNormalization, self).get_dim(name)

class BatchNormAccumulate(TrainingAlgorithm):
    """ TrainingAlgorithm that accumulates batchnorm parameters
    """
    def __init__(self, cg):
        self.cg = cg
        self.parameters = get_batchnorm_parameters(cg)
        self.inputs = cg.inputs
        self._input_names = [i.name for i in self.inputs]

    def initialize(self, **kwargs):
        logger.info("BatchNormAccumulate initializing")

        # get list of bricks
        bricks_seen = set()
        for p in self.parameters:
            brick = get_brick(p)
            if brick not in bricks_seen:
                bricks_seen.add(brick)

        # ensure all updates account for all bricks
        update_parameters = set()
        for b in bricks_seen:
            for var, update in b.updates.items():
                update_parameters.add(var)
            assert b.n.get_value() == 0

        if set(update_parameters) != set(self.parameters):
            raise ValueError("The updates and the parameters passed in do "
                "not match. This could be due to no applications or multiple applications "
                "found %d updates, and %d parameters"%(len(update_parameters), len(self.parameters)))

        updates = dict_union(*[b.updates for b in bricks_seen])

        logger.info("Compiling BatchNorm accumulate")
        self._func = theano.function(self.inputs, [], updates=updates, on_unused_input="warn")

        super(BatchNormAccumulate, self).initialize(**kwargs)

    def process_batch(self, batch):
        if not set(self._input_names).issubset((batch.keys())):
            raise ValueError("Invalid batch. Got sources: (%s), expected sources: (%s)"%(str(batch.keys()), str(self._input_names)))
        ordered_batch = [batch[v.name] for v in self.inputs]
        self._func(*ordered_batch)

def get_batchnorm_parameters(cg):
    """ Get the parameters marked with BATCHNORM_POPULATION
    Parameters
    ---------
    cg: `blocks.graph.ComputationGraph`
        computation graph to look through

    Returns
    -------
    variables: list
        list of variables
    """
    return VariableFilter(roles=[BATCHNORM_POPULATION])(cg.auxiliary_variables)

def infer_population(data_stream, model, n_batches):
    """ Sets the population parameters for a given model"""
    # construct a main loop with algorithm
    algorithm = BatchNormAccumulate(model)
    main_loop = MainLoop(
        algorithm=algorithm,
        data_stream=data_stream,
        model=model,
        extensions=[FinishAfter(after_n_batches=n_batches), ProgressBar()])
    main_loop.run()

    batchnorm_bricks = set([get_brick(p) for p in get_batchnorm_parameters(model)])
    for b in batchnorm_bricks:
        b.use_population = True

def get_batchnorm_parameter_dict(model):
    parameters = get_batchnorm_parameters(model)
    parameters_dict = OrderedDict()
    for p in parameters:
        name = get_parameter_name(p)
        parameters_dict[name] = p
    return parameters_dict

def get_batchnorm_parameter_values(model):
    bn_dict = get_batchnorm_parameter_dict(model)
    return {k:v.get_value() for k,v in bn_dict.items()}

def set_batchnorm_parameter_values(model, values_dict):
    bn_dict = get_batchnorm_parameter_dict(model)
    unknown = set(values_dict) - set(bn_dict)
    missing = set(bn_dict) - set(values_dict)
    if len(unknown):
        logger.error("unknown parameter names: {}\n".format(unknown))
    if len(missing):
        logger.error("missing values for parameters: {}\n".format(missing))

    for name, v in bn_dict.items():
        v.set_value(values_dict[name])
