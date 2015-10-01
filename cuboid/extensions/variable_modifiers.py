from blocks.extensions import SimpleExtension
from matplotlib import pylab as plt
from copy import deepcopy
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
        if self.parameter.name:
            self.main_loop.log.current_row["shared_variable_%s"%self.parameter.name] = new_value

    def debug_plot(self, max_iter=10000):
        x = np.linspace(0, max_iter, 10000)
        y = [self.compute_value(xx) for xx in x]
        plt.plot(x, y)
        plt.title(str(self))
        plt.xlabel("iterations")
        plt.ylabel("variable value")
        plt.show()


class ExponentialDecay(VariableModifier):
    """
    Modifies shared variable according to function
    v_0*e^(-n*exp_decay)

    Parameters
    ---------
    parameter: shared float
        variable updated by modifer
    exp_decay: float
        parameter which controls decay rate
    """
    def __init__(self, parameter, exp_decay):
        self.exp_decay = exp_decay
        super(ExponentialDecay, self).__init__(parameter)

    def compute_value(self, iterations):
        val = self.initial_value * np.exp(-iterations * self.exp_decay)
        return np.cast[np.float32](val)


class StepDecay(VariableModifier):
    """
    Modifies shared variable according to function
    v_0*multiplier^p where p is the number of steps passed in 
    the current iteration

    Parameters
    ---------
    parameter: shared float
        variable updated by modifer
    steps: list
        intervals to decrease the decay rate on
    multiplier: float
        rate to step the shared variable
    """
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


class InverseDecay(VariableModifier):
    """
    Modifies shared variable according to function
    v_0*(1.0+inv_decay*n)^-power

    Parameters
    ---------
    parameter: shared float
        variable updated by modifer
    inv_decay: float
        parameter which controls decay rate
    power: float
        parameter which controls decay rate
    """
    def __init__(self, parameter, inv_decay, power):
        self.inv_decay = inv_decay
        self.power = power
        super(InverseDecay, self).__init__(parameter)

    def compute_value(self, iterations):
        val = self.initial_value*(1.0+self.inv_decay*iterations)**-self.power
        return np.cast[np.float32](val)


class PolynomialDecay(VariableModifier):
    """
    Modifies shared variable according to function
    v_0*(1.0-n/max_iter)^power

    Parameters
    ---------
    parameter: shared float
        variable updated by modifer
    max_iter: float
        parameter which controls when shared variable
        should equal 0
    power: float
        parameter which controls decay rate
    """
    def __init__(self, parameter, max_iter, power):
        self.max_iter = max_iter
        self.power = power
        super(PolynomialDecay, self).__init__(parameter)

    def compute_value(self, iterations):
        if iterations >= self.max_iter:
            val = 0.0
        else:
            val = self.initial_value*(1.0 - 
                (iterations/np.float32(self.max_iter)))**self.power
        return np.cast[np.float32](val)


class SigmoidDecay(VariableModifier):
    """
    Modifies shared variable according to function
    v_0-v_0*(1.0/(1.0+e^(-sig_decay*n-step_size)))

    Parameters
    ---------
    parameter: shared float
        variable updated by modifer
    sig_decay: float
        parameter which controls decay rate
    step_size: float
        parameter which controls inflection point of decay rate
    """
    def __init__(self, parameter, sig_decay, step_size):
        self.sig_decay = sig_decay
        self.step_size = step_size
        super(SigmoidDecay, self).__init__(parameter)

    def compute_value(self, iterations):
        val = self.initial_value - self.initial_value * (1.0/(1.0 + 
            np.exp(-self.sig_decay * (iterations - self.step_size))))
        return np.cast[np.float32](val)


class LinearDecay(VariableModifier):
    """
    Modifies shared variable according to function
    v_0-(rate*n)

    Parameters
    ---------
    parameter: shared float
        variable updated by modifer
    rate: float
        parameter which controls decay rate
    clip_at_zero: boolean
        if set to true, shared variable will never 
        decrease to a value below 0.0
    """
    def __init__(self, parameter, rate, clip_at_zero=True):
        self.rate = rate
        self.clip_at_zero = clip_at_zero
        super(LinearDecay, self).__init__(parameter)

    def compute_value(self, iterations):
        val = self.initial_value-(self.rate*iterations)
        if self.clip_at_zero:
            val = max(0.0, val)
        return np.cast[np.float32](val)


class ComposeDecay(VariableModifier):
    """
    Modifies shared variable according to the 
        composition of variable modifiers
    Each modifier's initial value is set to the final 
        value of the predecessor modifier
    The first modifier is used from 
        iteration=0 to iteration=steps[0]

    Parameters
    ---------
    parameter: shared float
        variable updated by modifer
    variable_modifiers: list
        sequence of shared variable modifiers
    steps: list
        intervals to change which modifier is used on a 
        given iteration
    """
    def __init__(self, variable_modifiers, steps):
        self.variable_modifiers = deepcopy(variable_modifiers)
        self.steps = steps
        assert (len(steps) == (len(self.variable_modifiers)-1))
        assert all(self.variable_modifiers[0].parameter == 
            variable_modifier.parameter for 
            variable_modifier in self.variable_modifiers[1:])

        deltas = [steps[0]]
        for i in xrange(1, len(steps)):
            deltas.append(steps[i]-steps[i-1])
        for i in xrange(1, len(self.variable_modifiers)):
            self.variable_modifiers[i].initial_value = self.variable_modifiers[i-1].compute_value(deltas[i-1])

        super(ComposeDecay, self).__init__(variable_modifiers[0].parameter)

    def compute_value(self, iterations):
        step_index = 0
        for step in self.steps:
            if step > iterations:
                break
            step_index += 1
        if step_index > 0:
            return self.variable_modifiers[step_index].compute_value(iterations-self.steps[step_index-1])
        return self.variable_modifiers[step_index].compute_value(iterations)
