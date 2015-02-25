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

import shutil
import os
from blocks.dump import save_parameter_values
import datetime

class ExperimentSaver(TrainingExtension):
    def __init__(self, dest_directory, src_directory, **kwargs):
        super(ExperimentSaver, self).__init__(**kwargs)
        self.dest_directory = dest_directory
        self.src_directory = src_directory

        self.params_path = os.path.join(self.dest_directory, 'params')

    def params_path_for_epoch(self, i):
        return os.path.join(self.params_path, str(i))

    def before_training(self):
        if os.path.exists(self.dest_directory):
            time_string = datetime.datetime.now().strftime('%m-%d-%H-%M-%S') 
            move_to = self.dest_directory + time_string + "_backup"
            shutil.move(self.dest_directory, move_to)

        os.mkdir(self.dest_directory)
        src_path = os.path.join(self.dest_directory, 'src')

        os.mkdir(self.params_path)

        def ignore(path, names):
            # TODO actually manage paths correctly
            if path == self.dest_directory or path == './' + self.dest_directory:
                return names
            else:
                return []

        shutil.copytree(self.src_directory, src_path, ignore=ignore)

    def after_epoch(self):
        log = self.main_loop.log
        epoch_done = log.status.epochs_done
        params = self.main_loop.model.get_param_values()
        path = self.params_path_for_epoch(epoch_done)
        save_parameter_values(params, path)
