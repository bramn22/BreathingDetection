from stages.stage import StageInterface
import numpy as np


class Reshape(StageInterface):

    def __init__(self, outp_shape, **kwargs):
        super().__init__(**kwargs)
        self.outp_shape = outp_shape

    def _execute(self, inp, **kwargs):
        print("Received inp:", inp.shape)
        outp = np.reshape(inp, newshape=self.outp_shape)
        print("outp shape:", outp.shape)
        return outp

    def _visualize(self):
        pass


# class flatten_3d(StageInterface):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#
#     def _execute(self, inp, **kwargs):
#         print("Received inp:", inp.shape)
#         outp = np.reshape(inp, newshape=(len(inp), -1))
#         print("outp shape:", outp.shape)
#         return outp
#
#     def _visualize(self):
#         pass
#
#
# class expand_2d(StageInterface):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#
#     def _execute(self, inp, **kwargs):
#         print("Received inp:", inp.shape)
#         outp = np.reshape(inp, newshape=(len(inp), -1))
#         print("outp shape:", outp.shape)
#         return outp
#
#     def _visualize(self):
#         pass