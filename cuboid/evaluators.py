import theano
from blocks.monitoring.evaluators import AggregationBuffer
from blocks.utils import dict_subset
import numpy as np

from progressbar import ProgressBar, Percentage, ETA, Bar



class DatasetMapEvaluator(object):
    """
    A hack of a class to run a class and record all results over a datastream.
    This is useful for batch running a dataset though a model and saving the results.

    TODO: Should really be more blocks like with there existing framework and then tested
    """

    def __init__(self, variable, updates=None):
        # self.buffer_ is unused, just to grab names
        self.buffer_ = AggregationBuffer([variable])

        self.variables = [variable]
        self.updates = updates
        self._compile()
        self.accum = []

    def _compile(self):
        self._eval_fun = theano.function(
            self.buffer_.inputs, self.variables)

    def process_batch(self, batch):
        try:
            batch = dict_subset(batch, self.buffer_.input_names)
        except KeyError:
            reraise_as(
                "Not all data sources required for monitoring were"
                " provided. The list of required data sources:"
                " {}.".format(self.buffer_.input_names))
        self.accum.append(self._eval_fun(**batch))

    def get_aggregated_values(self):
        f = [a[0] for a in self.accum]
        return np.concatenate(f, axis=0)

    def evaluate(self, data_stream, printing=False, num_batches=None):
        if printing:
            widgets = [Percentage(), ' ', Bar(),
                       ' ', ETA(), ' ']
            progress = ProgressBar(widgets=widgets, maxval=num_batches).start()

        for i, batch in enumerate(data_stream.get_epoch_iterator(as_dict=True)):
            self.process_batch(batch)
            if printing:
                progress.update(i)

        return self.get_aggregated_values()
