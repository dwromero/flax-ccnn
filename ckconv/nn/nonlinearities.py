import flax.linen as nn

from functools import partial


class FunctionAsModule(nn.Module):
    function_name: str

    def setup(self):
        self.function = getattr(nn, self.function_name.lower())

    def __call__(self, x):
        return self.function(x)


GELU = partial(FunctionAsModule, function_name='GELU')
ReLU = partial(FunctionAsModule, function_name='ReLU')
