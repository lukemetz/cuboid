from blocks.utils import named_copy

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
