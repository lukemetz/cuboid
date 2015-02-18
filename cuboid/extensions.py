from blocks.extensions import SimpleExtension, TrainingExtension

from progressbar import ProgressBar, Percentage, ETA, Bar

class LogToFile(SimpleExtension):
    def __init__(self, file_name, **kwargs):
        kwargs.setdefault("before_first_epoch", True)
        kwargs.setdefault("on_resumption", True)
        kwargs.setdefault("after_training", True)
        kwargs.setdefault("after_every_epoch", True)
        kwargs.setdefault("on_interrupt", True)

        self.file_name = file_name

        super(LogToFile, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        self.main_loop.log.to_dataframe().to_csv(self.file_name)

class EpochProgress(TrainingExtension):
    def __init__(self, batch_per_epoch, **kwargs):
        super(EpochProgress, self).__init__(**kwargs)
        self.batch_per_epoch = batch_per_epoch

    def before_epoch(self):
        widgets = [Percentage(), ' ', Bar(),
                   ' ', ETA(), ' ']
        self.progress = ProgressBar(widgets=widgets, maxval=self.batch_per_epoch).start()
        self.on_batch = 0

    def before_batch(self, batch):
        self.progress.update(self.on_batch)
        self.on_batch += 1
