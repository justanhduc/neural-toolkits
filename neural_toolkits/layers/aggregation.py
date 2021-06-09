from .abstract import Module, MultiSingleInputModule, MultiMultiInputModule
from .. import utils

__all__ = ['Activation', 'Sum', 'SequentialSum', 'ConcurrentSum']


class Activation(Module):
    """
    Applies a non-linear function to the incoming input.

    Parameters
    ----------
    activation
        non-linear function to activate the linear result.
        It accepts any callable function
        as well as a recognizable ``str``.
        A list of possible ``str`` is in :func:`~neuralnet_pytorch.utils.function`.
    input_shape
        shape of the input tensor. Can be ``None``.
    kwargs
        extra keyword arguments to pass to activation.
    """

    def __init__(self, activation='relu', **kwargs):
        super().__init__()
        self.activation = utils.function(activation, **kwargs)

    def forward(self, input):
        return self.activation(input)

    def extra_repr(self):
        s = 'activation={}'.format(self.activation.__name__)
        return s


class Sum(MultiSingleInputModule):
    """
    Sums the outputs of multiple modules given an input tensor.
    A subclass of :class:`~neuralnet_pytorch.layers.MultiSingleInputModule`.

    See Also
    --------
    :class:`~neuralnet_pytorch.layers.MultiSingleInputModule`
    :class:`~neuralnet_pytorch.layers.MultiMultiInputModule`
    :class:`~neuralnet_pytorch.layers.SequentialSum`
    :class:`~neuralnet_pytorch.layers.ConcurrentSum`
    :class:`~neuralnet_pytorch.resizing.Cat`
    :class:`~neuralnet_pytorch.resizing.SequentialCat`
    :class:`~neuralnet_pytorch.resizing.ConcurrentCat`
    """

    def __init__(self, *modules_or_tensors):
        super().__init__(*modules_or_tensors)

    def forward(self, input):
        outputs = super().forward(input)
        return sum(outputs)


class SequentialSum(Sum):
    """
    Sums the intermediate outputs of multiple sequential modules given an input tensor.
    A subclass of :class:`~neuralnet_pytorch.layers.Sum`.

    See Also
    --------
    :class:`~neuralnet_pytorch.layers.MultiSingleInputModule`
    :class:`~neuralnet_pytorch.layers.MultiMultiInputModule`
    :class:`~neuralnet_pytorch.layers.Sum`
    :class:`~neuralnet_pytorch.layers.ConcurrentSum`
    :class:`~neuralnet_pytorch.resizing.Cat`
    :class:`~neuralnet_pytorch.resizing.SequentialCat`
    :class:`~neuralnet_pytorch.resizing.ConcurrentCat`
    """

    def __init__(self, *modules):
        super().__init__(*modules)

    def forward(self, input):
        outputs = []
        output = input
        for name, module in self.named_children():
            if name.startswith('tensor'):
                outputs.append(module())
            else:
                output = module(output)
                outputs.append(output)

        return sum(outputs)


class ConcurrentSum(MultiMultiInputModule):
    """
    Sums the outputs of multiple modules given input tensors.
    A subclass of :class:`~neuralnet_pytorch.layers.MultiMultiInputModule`.

    See Also
    --------
    :class:`~neuralnet_pytorch.layers.MultiSingleInputModule`
    :class:`~neuralnet_pytorch.layers.MultiMultiInputModule`
    :class:`~neuralnet_pytorch.layers.Sum`
    :class:`~neuralnet_pytorch.layers.SequentialSum`
    :class:`~neuralnet_pytorch.resizing.Cat`
    :class:`~neuralnet_pytorch.resizing.SequentialCat`
    :class:`~neuralnet_pytorch.resizing.ConcurrentCat`
    """

    def __init__(self, *modules_or_tensors):
        super().__init__(*modules_or_tensors)

    def forward(self, *input):
        outputs = super().forward(*input)
        return sum(outputs)
