import shutil
import os

from cuboid.dump import save_parameter_values, load_parameter_values
from cuboid.graph import (get_algorithm_parameters_values,
                          set_algorithm_parameters_values)

import datetime
import pandas as pd

from blocks.extensions import SimpleExtension, TrainingExtension
import time
import numpy as np
import cPickle
import logging

logger = logging.getLogger(__name__)


class LogToFile(SimpleExtension):
    """
    Logs training curve to file
    """
    def __init__(self, file_path, **kwargs):
        kwargs.setdefault("after_epoch", True)
        super(LogToFile, self).__init__(**kwargs)
        self.file_path = file_path

    def do(self, which_callback, *args):
        log = self.main_loop.log
        for k, v in log.status.items():
            if k[0] != '_':
                log.current_row[k] = v
        frame = pd.DataFrame.from_dict(log, orient='index')
        frame.index.name = "iterations"
        frame.to_csv(self.file_path)


class ExamplesPerSecond(TrainingExtension):
    def __init__(self, roll=10, batch_idx=0):
        self.example_accumulator = []
        self.last_time = time.time()
        self.times = []
        self.batches_seen = 0
        self.roll = roll
        self.batch_idx = batch_idx
        super(ExamplesPerSecond, self).__init__()

    def after_batch(self, batch):
        batch_size = len(batch.values()[self.batch_idx])
        self.example_accumulator.append(batch_size)
        if len(self.example_accumulator) > self.roll:
            self.example_accumulator.pop(0)

        new_time = time.time()
        examples_per_second = np.sum(self.example_accumulator) /\
                                    (np.sum(self.times))
        log = self.main_loop.log
        log.current_row['examples_per_second'] = examples_per_second
        self.times.append(new_time - self.last_time)
        self.last_time = new_time
        if len(self.times) > self.roll:
            self.times.pop(0)


class SavePoint(SimpleExtension):
    def __init__(self, dest_directory, **kwargs):
        super(SavePoint, self).__init__(**kwargs)
        self.dest_directory = dest_directory

        sub_folders = ['params', 'algorithm_params', 'logs']
        for s in sub_folders:
            path = os.path.join(self.dest_directory, s)
            if not os.path.exists(path):
                os.mkdir(path)

    def do(self, which_callback, *args):
        log = self.main_loop.log
        algorithm = self.main_loop.algorithm
        model = self.main_loop.model

        if which_callback == "after_epoch":
            done = log.status['epochs_done']
            prefix = "epoch"
        elif which_callback == "after_batch":
            done = log.status['iterations_done']
            prefix = "iterations"

        output_param_path = os.path.join(self.dest_directory, "params",
                                         "%s_%d.npz" % (prefix, done))

        output_algorithm_param_path = os.path.join(self.dest_directory,
                                                   "algorithm_params",
                                                   "%s_%d.npz" %
                                                   (prefix, done))

        output_log_path = os.path.join(self.dest_directory, "logs",
                                                            "%s_%d.pkl" %
                                                            (prefix, done))

        params = model.get_parameter_values()
        save_parameter_values(params, output_param_path)

        algorithm_params = get_algorithm_parameters_values(algorithm, model)

        save_parameter_values(algorithm_params, output_algorithm_param_path)

        cPickle.dump(log, open(output_log_path, 'w'))

        logger.info("Wrote new savepoint to (%s)" % self.dest_directory)


class Resume(SimpleExtension):
    def __init__(self, directory, place, **kwargs):
        self.directory = directory
        self.place = place
        self.has_resumed = False
        super(Resume, self).__init__(before_epoch=True)

    def do(self, which_callback, *args):
        if not self.has_resumed:
            self.has_resumed = True
            assert which_callback == "before_epoch"
            logger.info("loading from savepoint (%s/*/%s)" %
                        (self.directory, self.place))

            log_path = os.path.join(self.directory, "logs", self.place+".pkl")
            self.main_loop.log = cPickle.load(open(log_path))

            params_path = os.path.join(self.directory, "params",
                                       self.place+".npz")
            parameter_values = load_parameter_values(params_path)
            self.main_loop.model.set_parameter_values(parameter_values)

            algorithm_path = os.path.join(self.directory, "algorithm_params",
                                          self.place+".npz")
            algorithm_values = load_parameter_values(algorithm_path)
            set_algorithm_parameters_values(self.main_loop.algorithm,
                                            self.main_loop.model,
                                            algorithm_values)


class DirectoryCreator(SimpleExtension):
    def __init__(self, directory, **kwargs):
        if directory[-1] == "/":
            directory = directory[:-1]
        if os.path.exists(directory):
            time_string = datetime.datetime.now().strftime('%m-%d-%H-%M-%S')
            move_to = directory + time_string + "_backup"
            shutil.move(directory, move_to)
        os.mkdir(directory)
        super(DirectoryCreator, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        pass


class SourceSaver(SimpleExtension):
    """
    Save the source to a given folder

    Parameters
    ---------
    dest_directory: basestring
        Path to dump the experiment.
    src_directory: basestring
        Path to source to be copied.
    """

    def __init__(self, dest_directory, src_directory, **kwargs):
        self.dest_directory = dest_directory
        self.src_directory = src_directory

        self.params_path = os.path.join(self.dest_directory, 'params')
        self.write_src()

        super(SourceSaver, self).__init__(**kwargs)

    def write_src(self):
        src_path = os.path.join(self.dest_directory, 'src')
        os.mkdir(self.params_path)

        def ignore(path, names):
            # TODO actually manage paths correctly
            if path == self.dest_directory or\
               path == './' + self.dest_directory:

                return names
            else:
                return []

        shutil.copytree(self.src_directory, src_path, ignore=ignore)

    def do(self, which_callback, *args):
        pass


class UserFunc(SimpleExtension):
    """
    Run a user defined function while training.

    Parameters
    ---------
    func: callable,
        An instance of UserFunc will be passed in
    """
    def __init__(self, func, **kwargs):
        super(UserFunc, self).__init__(**kwargs)
        self.func = func

    def do(self, which_callback, *args):
        self.func(self)


class Profile(SimpleExtension):
    def __init__(self, **kwargs):
        kwargs.setdefault('before_first_epoch', True)
        kwargs.setdefault('after_epoch', True)
        super(Profile, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        current_row = self.main_loop.log.current_row
        profile = self.main_loop.profile.total

        total = sum(v for k, v in profile.items() if len(k) == 1)
        for name, val in profile.items():
            current_row["profile_"+"_".join(name)] = val
        current_row["profile_total"] = total
