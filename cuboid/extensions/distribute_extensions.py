from blocks.extensions import SimpleExtension
from distribute.wl import make_next_jobs_func

class DistributeUpdate(SimpleExtension):
    def __init__(self, worker, **kwargs):
        self.worker = worker
        kwargs.setdefault("every_n_epochs", 5)
        super(DistributeUpdate, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        epoch = self.main_loop.status['epochs_done']
        self.worker.commit_update("%d"%epoch)

class DistributeFinish(SimpleExtension):
    def __init__(self, worker, **kwargs):
        self.worker = worker
        kwargs.setdefault("after_training", True)

        super(DistributeFinish, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        epoch = self.main_loop.status['epochs_done']
        self.worker.commit_update("finished@ %d"%epoch)
        self.worker.finish_job()

class DistributeWhetlabFinish(SimpleExtension):
    def __init__(self, worker, whetlab_experiment, score_func, **kwargs):
        self.worker = worker
        self.experiment = whetlab_experiment
        self.score_func = score_func

        kwargs.setdefault("after_training", True)

        super(DistributeWhetlabFinish, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        epoch = self.main_loop.status['epochs_done']
        self.worker.commit_update("finished@ %d"%epoch)
        finished_job = self.worker.running_job
        self.worker.finish_job()

        print "Calculating score"
        score = self.score_func(self.main_loop)
        print "Sending score for job %s of value %f"%(finished_job, score)
        self.experiment.update_by_result_id(finished_job, score)
