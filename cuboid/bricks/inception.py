from blocks.bricks import Initializable
from blocks.bricks.base import lazy, application
import theano.tensor as T


class ConcatBricks(Initializable):
    @lazy(allocation=['input_dim'])
    def __init__(self, bricks, input_dim, axis=1, **kwargs):
        super(ConcatBricks, self).__init__(**kwargs)
        self.children = bricks
        self.axis = axis
        self.input_dim = input_dim

    @application(inputs=["input_"], outputs=["output"])
    def apply(self, input_):
        results = [child.apply(input_) for child in self.children]
        return T.concatenate(results, axis=self.axis)

    def get_dim(self, name):
        if name == "input_":
            return self.input_dim
        if name == "output":
            dims = [child.get_dim(name) for child in self.children]
            if not all([type(d) == type(dims[0]) for d in dims]):
                raise ValueError("Types returned by children are not"
                                 " all the same")
            indx = self.axis - 1
            if indx < 0:
                return sum(dims)
            else:
                concat_axis = sum([d[indx] for d in dims])
                ret = list(dims[0][:])
                ret[indx] = concat_axis
                return tuple(ret)
        return super(ConcatBricks, self).get_dim(name)


# class Inception(ConcatBricks):
#     def __init__(self, input_dim,
#                  n_1x1,
#                  n_3x3_reduce, n_3x3,
#                  n_5x5_reduce, n_5x5,
#                  n_pool_proj,
#                  **kwargs):
#         super(Inception,
#         self.n_1x1 = n_1x1
#         self.n_3x3_reduct = n_3x3_reduce
#         self.n_3x3 = n_3x3
#         self.n_5x5_reduce = n_5x5_reduce
#         self.n_pool_proj = n_pool_proj
