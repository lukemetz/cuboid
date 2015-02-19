from blocks.datasets.streams import DataStreamWrapper
import futures

from multiprocessing import Process, Queue


class BackgroudProcess(object):
    def __init__(self, data_stream, max_batches=100):
        self.data_stream = data_stream
        self.batches = Queue(max_batches)
        self.run_background = True

    def main(self):
        while self.run_background:
            iterator = self.data_stream.get_epoch_iterator()
            for batch in iterator:
                self.batches.put(batch)

            self.batches.put(StopIteration)

    def get_next_data(self):
        return self.batches.get()

    def stop_background(self):
        self.run_background = False

class DataStreamBackground(DataStreamWrapper):
    """
    Wrapper to make the input stream run in a background thread.
    Always have the next batch ready.

    """

    def __init__(self, data_stream, max_store=100):
        super(DataStreamBackground, self).__init__(data_stream)
        self.background = BackgroudProcess(data_stream)
        self.proc = Process(target=self.background.main)
        self.proc.start()

    def __del__(self):
        self.background.stop_background()
        self.proc.join()

    def get_data(self, request=None):
        if request is not None:
            raise ValueError

        data = self.background.get_next_data()
        if data == StopIteration:
            raise StopIteration
        else:
            return data
