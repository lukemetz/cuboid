from blocks.bricks.base import lazy, application
from blocks.bricks import Linear, Initializable
import numpy as np
from blocks.initialization import NdarrayInitialization


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
            raise ValueError("Unexpected number of dims (%s)"%(str(shape)))

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
