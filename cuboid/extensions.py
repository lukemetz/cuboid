from blocks.extensions import SimpleExtension

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
