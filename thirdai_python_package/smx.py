import thirdai._thirdai.smx
from thirdai._thirdai.smx import *

__all__ = []
__all__.extend(dir(thirdai._thirdai.smx))


class Module(thirdai._thirdai.smx._Module):
    def __init__(self):
        super().__init__()

    def forward(self, *inputs):
        raise NotImplementedError()

    def _forward(self, inputs):
        out = self.forward(*inputs)
        # The module interface expects a list of Variables returned as output.
        if isinstance(out, thirdai._thirdai.smx.Variable):
            return [out]
        return out

    def __call__(self, *inputs):
        return self.forward(*inputs)

    def __setattr__(self, name, value):
        if isinstance(value, thirdai._thirdai.smx._Module):
            self.register_module(name, value)
        elif isinstance(value, thirdai._thirdai.smx.Variable):
            self.register_parameter(name, value)

        super().__setattr__(name, value)
