from blocks.utils import named_copy
from blocks.filter import VariableFilter
from blocks.roles import ALGORITHM_STATE
from blocks.graph import ComputationGraph
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)

def parameter_stats(cg):
    observables = []
    for name,parameter in cg.get_parameter_dict().items():
        observables.append(named_copy(
            parameter.norm(2), name + "_norm"))
        observables.append(named_copy(
            parameter.mean(), name + "_mean"))
        observables.append(named_copy(
            parameter.var(), name + "_var"))
    return observables

def gradient_stats(cg, algorithm):
    observables = []
    for name,parameter in cg.get_parameter_dict().items():
        observables.append(named_copy(
            algorithm.gradients[parameter].norm(2), name + "_grad_norm"))
    return observables

def step_stats(cg, algorithm):
    observables = []
    for name,parameter in cg.get_parameter_dict().items():
        observables.append(named_copy(
            algorithm.steps[parameter].norm(2), name + "_step_norm"))
    return observables

def get_algorithm_parameters_dict(algorithm, model):
    name_to_var = model.get_parameter_dict()
    var_to_name = {v:k for k,v in name_to_var.items()}

    output_dict = dict()

    for val, update in algorithm.steps.items():
        cg = ComputationGraph([update])
        shared_to_save = VariableFilter(roles=[ALGORITHM_STATE])(cg)

        parent_name = var_to_name[val]
        for k in shared_to_save:
            output_dict[parent_name+"/"+k.name] = k
    return output_dict

def get_algorithm_parameters_values(algorithm, model):
    dd = get_algorithm_parameters_dict(algorithm, model)
    out = dict()
    for key,var in dd.items():
        out[key] = var.get_value()
    return out

def set_algorithm_parameters_values(algorithm, model, values_dict):
    parameters_dict = get_algorithm_parameters_dict(algorithm, model)
    unknown = set(values_dict) - set(parameters_dict)
    missing = set(parameters_dict) - set(values_dict)
    if len(unknown):
        logger.error("unknown parameter names: {}\n".format(unknown))
    if len(missing):
        logger.error("missing values for parameters: {}\n".format(missing))

    for name, value in values_dict.items():
        if name in parameters_dict:
            parameters_dict[name].set_value(value)
