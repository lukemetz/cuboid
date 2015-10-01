from fuel.transformers import Transformer


class DropSources(Transformer):
    """Drops some sources from a stream

    Parameters
    ----------
    data_stream : :class:`DataStream` or :class:`Transformer`.
        The data stream.
    sources: list
        A list of sources to drop
    """
    def __init__(self, data_stream, sources):
        super(DropSources, self).__init__(data_stream)
        old_sources = list(self.data_stream.sources)
        self.mask = [True for _ in old_sources]

        cur_sources = old_sources[:]

        for i, s in enumerate(sources):
            if s not in cur_sources:
                raise KeyError("%s not in the sources of the stream" % s)
            else:
                cur_sources.remove(s)
                self.mask[old_sources.index(s)] = False
        self.sources = tuple(cur_sources)

    def get_data(self, request=None):
        if request is not None:
            raise ValueError
        data = next(self.child_epoch_iterator)
        new_data = tuple([source for source, mask in zip(data, self.mask)
                          if mask])
        return new_data
