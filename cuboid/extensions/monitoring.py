from blocks.extensions import SimpleExtension
from blocks.extensions.monitoring import MonitoringExtension
from cuboid.evaluators import DataStreamEvaluator
from sklearn.metrics import roc_auc_score, confusion_matrix
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)


class AUCMonitor(SimpleExtension, MonitoringExtension):
    """
    Monitors auc scores on a datastream

    Parameters
    ---------
    data_stream: instance of :class:`.DataStream`
        The datas_tream to monitor on
    probs: matrix
        predicted value probability distribution used to compute auc
    labels: vector
        true values used to compute auc
    """

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
    """
    Monitors accuracy scores on a datastream on a per class basis

    Parameters
    ---------
    data_stream: instance of :class:`.DataStream`
        The datas_tream to monitor on
    prediction: vector
        predicted values to compute accuracy
    targets: vector
        true values used to compute accuracy
    label_i_to_c: dict
        mapping between a class label (integer) and its semantic name (string)

    Notes
    ---------
    Example usage
    >>> perclass_am = PerClassAccuracyMonitor(datastream,
                                prediction=numpy.argmax(probs, axis=1),
                                targets=targets.ravel(),
                                label_i_to_c=label_i_to_c)

    """

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
                        [('perclass_accuracy_%s'%label, accuracy) 
                        for (label,accuracy) in 
                        perclass_accuracy_map.iteritems()])


class ConfusionMatrixMonitor(SimpleExtension, MonitoringExtension):
    """
    Monitors confusion matrix on a datastream. Saves matrices to a directory as
    an npz file

    Parameters
    ---------
    data_stream: instance of :class:`.DataStream`
        The datas_tream to monitor on
    prediction: vector
        predicted values to compute accuracy
    targets: vector
        true values used to compute accuracy
    dest_directory: string
        output directory for storing confusion matrices

    Notes
    ---------
    Example usage
    >>> confusion_m = ConfusionMatrixMonitor(datastream,
                                prediction=numpy.argmax(probs, axis=1),
                                targets=targets.ravel(),
                                dest_directory='confusionMatrices')

    """

    def __init__(self, data_stream, prediction, targets, dest_directory, **kwargs):
        self.data_stream = data_stream
        self.evaluator = DataStreamEvaluator([prediction.copy('prediction'),
                                             targets.copy('targets')])
        self.dest_directory = dest_directory

        if not os.path.exists(self.dest_directory):
            os.mkdir(self.dest_directory)

        if prediction.ndim != 1 or targets.ndim != 1:
            raise ValueError("targets and predictions must be 1d vectors")
        logger.info("compiling confusion matrix monitor")
        self.evaluator._compile()
        super(ConfusionMatrixMonitor, self).__init__(**kwargs)

    def do(self, callback_name, *args):
        dd = self.evaluator.evaluate(self.data_stream)
        predictions = dd['prediction']
        targets = dd['targets']
        confusion = confusion_matrix(targets, predictions)
        confusion = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]

        log = self.main_loop.log
        if callback_name == "after_epoch":
            done = log.status['epochs_done']
            prefix = "epoch"
        elif callback_name == "after_batch":
            done = log.status['iterations_done']
            prefix = "iterations"
        else:
            logging.warn('confusion matrix extension won\'t save without \
                         setting a callback interval e.g. every_n_batches=3')

        output_path = os.path.join(self.dest_directory,
                                   "confusion_%s_%d.npz" % (prefix, done))
        with open(output_path, 'wb') as f:
            np.save(f, confusion)
