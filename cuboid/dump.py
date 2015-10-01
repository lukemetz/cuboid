# Taken directly from older version of blocks with many deletions
# and modifications.

import logging

import numpy

logger = logging.getLogger(__name__)


def save_parameter_values(param_values, path):
    """Compactly save parameter values.

    This is a thin wrapper over `numpy.savez`. It deals with
    `numpy`'s vulnerability to slashes in file names.

    Parameters
    ----------
    param_values : dict of (parameter name, numpy array)
        The parameter values.
    path : str of file
        The destination for saving.

    """
    param_values = {name.replace("/", "-"): param
                    for name, param in param_values.items()}
    if len(param_values.items()) == 0:
        numpy.savez(path, __placeholder=None)
    else:
        numpy.savez(path, **param_values)


def load_parameter_values(path):
    """Load parameter values saved by :func:`save_parameters`.

    This is a thin wrapper over `numpy.load`. It deals with
    `numpy`'s vulnerability to slashes in file names.

    Parameters
    ----------
    path : str or file
        The source for loading from.

    Returns
    -------
    A dictionary of (parameter name, numpy array) pairs.

    """
    source = numpy.load(path)
    param_values = {name.replace("-", "/"): value
                    for name, value in source.items()
                    if name != "__placeholder"}
    source.close()
    return param_values
