from blocks.extensions import SimpleExtension
from blocks.extensions.monitoring import MonitoringExtension
from cuboid.evaluators import DataStreamEvaluator
from sklearn.metrics import roc_auc_score
import numpy as np
import logging

logger = logging.getLogger(__name__)


class AUCMonitor(SimpleExtension, MonitoringExtension):
    def __init__(self, data_stream, probs, labels, **kwargs):
        self.data_stream = data_stream
        self.evaluator = DataStreamEvaluator([probs.copy('probs'),
                                              labels.copy('targets')])
        logger.info("compiling auc logger")
        self.evaluator._compile()
        super(AUCMonitor, self).__init__(**kwargs)

    def do(self, callback_name, *args):
        dd = self.evaluator.evaluate(self.data_stream)
        prob = dd['probs']
        targets = dd['targets']
        score = roc_auc_score(targets, prob)
        self.add_records(self.main_loop.log, [('auc', score)])

class PerClassAccuracyMonitor(SimpleExtension, MonitoringExtension):
    def __init__(self, data_stream, prediction, targets, label_i_to_c, **kwargs):
        self.data_stream = data_stream
        self.label_i_to_c = label_i_to_c
        self.evaluator = DataStreamEvaluator([prediction.copy('prediction'), 
                                              targets.copy('targets')])
        if prediction.ndim != 1 or targets.ndim != 1:
            raise ValueError("targets and predictions must be 1d vectors")
        logger.info("compiling perclass accuracy logger")
        self.evaluator._compile()
        super(PerClassAccuracyMonitor, self).__init__(**kwargs)

    def do(self, callback_name, *args):
        dd = self.evaluator.evaluate(self.data_stream)
        prediction = dd['prediction']
        targets = dd['targets']
        perclass_accuracy_map = {}
        prediction = prediction.reshape((-1,1))
        targets = targets.reshape((-1,1))
        correct = (prediction==targets)
        for target in sorted(self.label_i_to_c.keys()):
            perclass_accuracy_map[self.label_i_to_c[target]] = (
                np.sum(correct[targets==target]) / 
                np.float32(np.sum((targets==target))))
        self.add_records(self.main_loop.log, 
                        [('perclass accuracy_%s'%label, accuracy) 
                        for (label,accuracy) in 
                        perclass_accuracy_map.iteritems()])
