from blocks.extensions import SimpleExtension
from matplotlib import pylab as plt
import numpy as np


class VariableModifier(SimpleExtension):
    def __init__(self, parameter):
        self.parameter = parameter
        kwargs = dict()
        kwargs.setdefault("after_batch", True)
        self.initial_value = np.copy(parameter.get_value())
        super(VariableModifier, self).__init__(**kwargs)

    def compute_value(self, iterations):
        pass

    def do(self, which_callback, *args):
        iterations_done = self.main_loop.log.status['iterations_done']
        new_value = self.compute_value(iterations_done)
        self.parameter.set_value(new_value)

    def debug_plot(self, max_iter=10000):
        x = np.linspace(0, max_iter, 10000)
        y = [self.compute_value(xx) for xx in x]
        plt.plot(x, y)
        plt.title(str(self))
        plt.xlabel("iterations")
        plt.ylabel("variable value")
        plt.show()


class ExponentialDecay(VariableModifier):
    def __init__(self, parameter, exp_decay):
        self.exp_decay = exp_decay
        super(ExponentialDecay, self).__init__(parameter)

    def compute_value(self, iterations):
        val = self.initial_value * np.exp(-iterations * self.exp_decay)
        return np.cast[np.float32](val)


class StepDecay(VariableModifier):
    def __init__(self, parameter, steps, multiplier=0.5):
        self.steps = steps
        self.multiplier = multiplier
        super(StepDecay, self).__init__(parameter)

    def compute_value(self, iterations):
        c = np.copy(self.initial_value)
        for s in self.steps:
            if iterations >= s:
                c *= self.multiplier
        return np.cast[np.float32](c)
