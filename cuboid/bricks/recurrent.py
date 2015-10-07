from blocks.bricks.base import lazy, application
from blocks.bricks import Linear, Initializable, Tanh, Logistic
from blocks.bricks.recurrent import BaseRecurrent, recurrent
import numpy as np
from blocks.initialization import NdarrayInitialization
from blocks.utils import shared_floatx_nans, shared_floatx_zeros
from blocks.roles import add_role, WEIGHT, INITIAL_STATE
import theano.tensor as T


class GruInitialization(NdarrayInitialization):
    def __init__(self, reset_init=None, update_init=None, **kwargs):
        super(GruInitialization, self).__init__(**kwargs)
        self.reset_init = reset_init
        self.update_init = update_init

    def generate(self, rng, shape):
        if len(shape) == 1:
            assert shape[0] % 2 == 0
            half_shape = (shape[0]/2, )
        elif len(shape) == 2:
            assert shape[1] % 2 == 0
            half_shape = (shape[0], shape[1]/2)
        else:
            raise ValueError("Unexpected number of dims (%s)" % (str(shape)))

            raise ValueError("Gru not initialized. missing weights or biases")

        updates = self.update_init.generate(rng, half_shape)
        resets = self.reset_init.generate(rng, half_shape)
        return np.hstack((updates, resets))


class GatedRecurrentFork(Initializable):
    @lazy(allocation=['input_dim', 'hidden_dim'])
    def __init__(self, input_dim, hidden_dim,
                 inputs_weights_init=None,
                 inputs_biases_init=None,
                 reset_weights_init=None,
                 reset_biases_init=None,
                 update_weights_init=None,
                 update_biases_init=None,
                 **kwargs):
        super(GatedRecurrentFork, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.inputs_weights_init = inputs_weights_init
        self.inputs_biases_init = inputs_biases_init

        self.reset_weights_init = reset_weights_init
        self.reset_biases_init = reset_biases_init

        self.update_weights_init = update_weights_init
        self.update_biases_init = update_biases_init

        self.input_to_inputs = Linear(input_dim=input_dim,
                                      output_dim=self.hidden_dim,
                                      name="input_to_inputs")
        self.input_to_gate_inputs = Linear(input_dim=input_dim,
                                           output_dim=self.hidden_dim*2,
                                           name="input_to_gate_inputs")

        self.children = [self.input_to_inputs, self.input_to_gate_inputs]

    def _initialize(self):
        # super(GatedRecurrentFork, self)._initialize()
        pass

    def _push_allocation_config(self):
        super(GatedRecurrentFork, self)._push_allocation_config()

    def _push_initialization_config(self):
        super(GatedRecurrentFork, self)._push_initialization_config()

        if self.inputs_weights_init:
            self.input_to_inputs.weights_init = self.inputs_weights_init
        if self.inputs_biases_init:
            self.input_to_inputs.biases_init = self.inputs_biases_init

        init = GruInitialization(reset_init=self.weights_init,
                                 update_init=self.weights_init)

        if self.update_weights_init:
            init.update_init = self.update_weights_init
        if self.reset_weights_init:
            init.reset_init = self.reset_weights_init
        self.input_to_gate_inputs.weights_init = init

        init = GruInitialization(reset_init=self.biases_init,
                                 update_init=self.biases_init)

        if self.update_biases_init:
            init.update_init = self.update_biases_init
        if self.reset_biases_init:
            init.reset_init = self.reset_biases_init
        self.input_to_gate_inputs.biases_init = init

    @application(inputs=['input_'], outputs=['inputs', 'gate_inputs'])
    def apply(self, input_):
        inputs = self.input_to_inputs.apply(input_)
        gate_inputs = self.input_to_gate_inputs.apply(input_)

        return inputs, gate_inputs


class MUT3(BaseRecurrent, Initializable):
    """ Mutation 3 as described by [1]

    Reference
    ---------
    ..[1] http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf
    """

    @lazy(allocation=['dim'])
    def __init__(self, dim, activation=None, gate_activation=None,
                 **kwargs):
        super(MUT3, self).__init__(**kwargs)
        self.dim = dim

        if not activation:
            activation = Tanh()
        if not gate_activation:
            gate_activation = Logistic()
        self.activation = activation
        self.gate_activation = gate_activation

        self.children = [activation, gate_activation]

    @property
    def state_to_state(self):
        return self.parameters[0]

    @property
    def state_to_z(self):
        return self.parameters[1]

    @property
    def state_to_r(self):
        return self.parameters[2]

    def get_dim(self, name):
        if name == 'mask':
            return 0
        if name in ['inputs', 'states']:
            return self.dim
        if name in ['z', 'r']:
            return self.dim
        return super(MUT3, self).get_dim(name)

    def _allocate(self):
        self.parameters.append(shared_floatx_nans((self.dim, self.dim),
                               name='state_to_state'))
        self.parameters.append(shared_floatx_nans((self.dim, self.dim),
                               name='state_to_z'))
        self.parameters.append(shared_floatx_nans((self.dim, self.dim),
                               name='state_to_r'))
        self.parameters.append(shared_floatx_zeros((self.dim,),
                               name="initial_state"))
        for i in range(3):
            if self.parameters[i]:
                add_role(self.parameters[i], WEIGHT)
        add_role(self.parameters[3], INITIAL_STATE)

    def _initialize(self):
        self.weights_init.initialize(self.state_to_state, self.rng)
        self.weights_init.initialize(self.state_to_r, self.rng)
        self.weights_init.initialize(self.state_to_z, self.rng)

    @recurrent(sequences=['mask', 'inputs', 'z_inputs', 'r_inputs'],
               states=['states'], outputs=['states'], contexts=[])
    def apply(self, inputs, z_inputs, r_inputs, states, mask=None):
        z = self.gate_activation.apply(
            states.dot(self.activation.apply(self.state_to_z)) + z_inputs)
        r = self.gate_activation.apply(
            states.dot(self.state_to_r) + r_inputs)
        states_reset = states * r
        next_states = self.activation.apply(
            states_reset.dot(self.state_to_state) + inputs)
        next_states = (next_states * (1 - z) + states * z)
        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)
        return next_states

    @application(outputs=apply.states)
    def initial_states(self, batch_size, *args, **kwargs):
        return [T.repeat(self.parameters[3][None, :], batch_size, 0)]


class GRU(BaseRecurrent, Initializable):
    @lazy(allocation=['dim'])
    def __init__(self, dim, activation=None, gate_activation=None,
                 **kwargs):
        super(GRU, self).__init__(**kwargs)
        self.dim = dim

        if not activation:
            activation = Tanh()
        if not gate_activation:
            gate_activation = Logistic()
        self.activation = activation
        self.gate_activation = gate_activation

        self.children = [activation, gate_activation]

    @property
    def state_to_state(self):
        return self.parameters[0]

    @property
    def state_to_z(self):
        return self.parameters[1]

    @property
    def state_to_r(self):
        return self.parameters[2]

    def get_dim(self, name):
        if name == 'mask':
            return 0
        if name in ['inputs', 'states']:
            return self.dim
        if name in ['z', 'r']:
            return self.dim
        return super(GRU, self).get_dim(name)

    def _allocate(self):
        self.parameters.append(shared_floatx_nans((self.dim, self.dim),
                               name='state_to_state'))
        self.parameters.append(shared_floatx_nans((self.dim, self.dim),
                               name='state_to_z'))
        self.parameters.append(shared_floatx_nans((self.dim, self.dim),
                               name='state_to_r'))
        self.parameters.append(shared_floatx_zeros((self.dim,),
                               name="initial_state"))
        for i in range(3):
            if self.parameters[i]:
                add_role(self.parameters[i], WEIGHT)
        add_role(self.parameters[3], INITIAL_STATE)

    def _initialize(self):
        self.weights_init.initialize(self.state_to_state, self.rng)
        self.weights_init.initialize(self.state_to_r, self.rng)
        self.weights_init.initialize(self.state_to_z, self.rng)

    @recurrent(sequences=['mask', 'inputs', 'z_inputs', 'r_inputs'],
               states=['states'], outputs=['states'], contexts=[])
    def apply(self, inputs, z_inputs, r_inputs, states, mask=None):
        z = self.gate_activation.apply(
            states.dot(self.state_to_z) + z_inputs)
        r = self.gate_activation.apply(
            states.dot(self.state_to_r) + r_inputs)
        states_reset = states * r
        next_states = self.activation.apply(
            states_reset.dot(self.state_to_state) + inputs)
        next_states = (next_states * (1 - z) + states * z)
        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)
        return next_states

    @application(outputs=apply.states)
    def initial_states(self, batch_size, *args, **kwargs):
        return [T.repeat(self.parameters[3][None, :], batch_size, 0)]
