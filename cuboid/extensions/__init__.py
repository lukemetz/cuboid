import shutil
import os
from cuboid.dump import save_parameter_values
import datetime
import pandas as pd
import ipdb

from blocks.extensions import SimpleExtension, TrainingExtension
import json

class ExperimentSaver(SimpleExtension):
    """
    Save a given experiment to a file
    * dump the current source directory
    * save parameters
    * dump the training logs

    Parameters
    ---------
    dest_directory: basestring
        Path to dump the experiment.
    src_directory: basestring
        Path to source to be copied.
    config: dict
        python dictionary representing configs
    """

    def __init__(self, dest_directory, src_directory, config={}, **kwargs):
        self.dest_directory = dest_directory
        self.src_directory = src_directory
        self.config = config

        self.params_path = os.path.join(self.dest_directory, 'params')
        self.log_path = os.path.join(self.dest_directory, 'log.csv')

        self.write_src()
        self.write_config()

        super(ExperimentSaver, self).__init__(**kwargs)

    def params_path_for_epoch(self, i):
        return os.path.join(self.params_path, str(i))

    def write_src(self):
        # Don't overwrite anything, move it to a backup folder
        if os.path.exists(self.dest_directory):
            import ipdb
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

    def write_config(self):
        json.dump(self.config, open(os.path.join(self.dest_directory, 'config.json'), 'w+'))

    def do(self, which_callback, *args):
        log = self.main_loop.log
        epoch_done = log.status['epochs_done']
        params = self.main_loop.model.get_parameter_values()
        path = self.params_path_for_epoch(epoch_done)
        save_parameter_values(params, path)

        pd.DataFrame.from_dict(log, orient='index').to_csv(self.log_path)
