from blocks.extensions import SimpleExtension
from blocks.extensions.monitoring import MonitoringExtension
from cuboid.evaluators import DataStreamEvaluator
from sklearn.metrics import roc_auc_score

class AUCMonitor(SimpleExtension, MonitoringExtension):
    def __init__(self, data_stream, probs, labels, **kwargs):
        self.data_stream = data_stream
        self.evaluator = DataStreamEvaluator([probs.copy('probs'), labels.copy('targets')])
        print "compiling auc logger"
        self.evaluator._compile()
        super(AUCMonitor, self).__init__(**kwargs)

    def do(self, callback_name, *args):
        dd = self.evaluator.evaluate(self.data_stream)
        prob = dd['probs']
        targets = dd['targets']
        score = roc_auc_score(targets, prob)
        self.add_records(self.main_loop.log, [('auc', score)])
