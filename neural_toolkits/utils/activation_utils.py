import torch as T
import torch.nn.functional as F
from functools import partial, update_wrapper
import inspect


def linear(x: T.Tensor):
    """
    Linear activation.
    """

    return x


class Swish(T.nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace
        self.beta = T.nn.Parameter(T.ones(1), requires_grad=True)

    def forward(self, input):
        return F.silu(self.beta * input, inplace=self.inplace) / (self.beta + 1e-8)


act = {
    'relu': F.relu,
    'linear': linear,
    None: linear,
    'lrelu': F.leaky_relu,
    'tanh': F.tanh,
    'sigmoid': F.sigmoid,
    'elu': F.elu,
    'softmax': F.softmax,
    'selu': F.selu,
    'silu': F.silu,
    'glu': F.glu,
    'prelu': T.nn.PReLU,
    'swish': Swish
}


def function(activation, **kwargs):
    """
    returns the `activation`. Can be ``str`` or ``callable``.
    For ``str``, possible choices are
    ``None``, ``'linear'``, ``'relu'``, ``'lrelu'``,
    ``'tanh'``, ``'sigmoid'``, ``'elu'``, ``'softmax'``,
    and ``'selu'``.

    :param activation:
        name of the activation function.
    :return:
        activation function
    """
    if isinstance(activation, str) or activation is None:
        activation = act[activation]

    try:
        arguments = inspect.BoundArguments(inspect.signature(activation), kwargs)
        if inspect.isclass(activation) and issubclass(activation, T.nn.Module):
            func = activation(**arguments.kwargs)
            setattr(func, '__name__', activation.__name__)
        else:
            func = partial(activation, **arguments.kwargs)
            update_wrapper(func, activation)
            if isinstance(activation, T.nn.Module):
                setattr(func, '__name__', activation._get_name())
    except ValueError:
        func = partial(activation, **kwargs)
        update_wrapper(func, activation)
        if isinstance(activation, T.nn.Module):
            setattr(func, '__name__', activation._get_name())

    return func
