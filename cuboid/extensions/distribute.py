from blocks.extensions import SimpleExtension

class DistributeUpdateAndFinish(SimpleExtension):
    def __init__(self, worker, **kwargs):
        self.worker = worker
        kwargs.setdefault("every_n_epochs", 5)
        kwargs.setdefault("after_training", True)

        super(DistributeUpdateAndFinish, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        epoch = self.main_loop.status['epochs_done']
        if which_callback == "after_training":
            self.worker.commit_changes("finished@ %d"%epoch)
            self.worker.finish_job()
        else:
            self.worker.commit_changes("%d"%epoch)
