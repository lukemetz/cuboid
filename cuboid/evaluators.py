from blocks.graph import ComputationGraph
from collections import OrderedDict
import theano
from blocks.utils import dict_subset
import numpy as np

class DataStreamEvaluator(object):
    """ Evaluate a datastream for paticular variables.
    Unlike blocks built in DataSetEvaluator, this does no accumulation
    instead it just returns a batch concatenated list of values.
    This is useful wanting the predictions themselves instead of
    just performance.

    Parameters
    ---------
    variables: list
        list of theano variables to monitor
    """
    def __init__(self, variables):
        self._computation_graph = ComputationGraph(variables)
        self._func = None
        self.inputs = self._computation_graph.inputs
        self.outputs = self._computation_graph.outputs

    def evaluate(self, data_stream):
        """ Runs the evaluation on a data_stream.

        Parameters
        ----------
        data_stream: `fuel.data_stream.DataStream`
            datastream to run computation on
        """
        accumulate_dict = OrderedDict([(var.name, []) for var in self.outputs])
        if self._func == None:
            self._compile()
        for batch in data_stream.get_epoch_iterator(as_dict=True):
            self.process_batch(batch, accumulate_dict)

        for key,value in accumulate_dict.items():
            accumulate_dict[key] = np.concatenate(value, axis=0)
        return accumulate_dict

    def process_batch(self, batch, accumulate_dict):
        try:
            input_names = [v.name for v in self.inputs]
            batch = dict_subset(batch, input_names)
        except KeyError:
            reraise_as(
                "Not all data sources required for monitoring were"
                " provided. The list of required data sources:"
                " {}.".format(input_names))
        results_list = self._func(**batch)
        output_names = [v.name for v in self.outputs]
        for name,res in zip(output_names, results_list):
            accumulate_dict[name].append(res)

    def _compile(self):
        self._func = theano.function(self.inputs, self.outputs)
