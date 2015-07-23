def get_parameters_from_brick(brick):
    """
    recursivly go through a brick and get parameters

    Parameters
    ----------
    brick: ~blocks.brick.base.Brick

    Returns
    -------
    parameters: list ~theano.tensor.TensorVariable
    """

    parameters = brick.parameters[:]
    for c in brick.children:
        parameters.extend(get_parameters_from_brick(c))
    return parameters
