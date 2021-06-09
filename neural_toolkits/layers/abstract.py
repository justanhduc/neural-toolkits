from collections import OrderedDict

import torch as T
import torch.nn as nn

from .. import utils
from ..utils import add_custom_repr

__all__ = ['wrapper', 'Sequential', 'Lambda', 'Module', 'MultiSingleInputModule', 'MultiMultiInputModule',
           'SingleMultiInputModule', 'Eye']


class _LayerMethod:
    """
    This mixin class contains various attributes to extend :mod:`torch` modules.
    """

    @property
    def trainable(self):
        """
        Return a tuple of all parameters with :attr:`requires_grad` set to `True`.
        """

        assert not hasattr(super(), 'trainable')
        params = []
        if hasattr(self, 'parameters'):
            params = [p for p in self.parameters() if p.requires_grad]
        return tuple(params)

    @property
    def regularizable(self):
        """
        Returns a tuple of parameters to be regularized.
        """

        assert not hasattr(super(), 'regularizable')
        params = []
        if hasattr(self, 'weight'):
            if self.weight.requires_grad:
                params += [self.weight]

        for m in list(self.children()):
            if hasattr(m, 'regularizable'):
                params.extend(m.regularizable)

        return tuple(params)

    def save(self, param_file):
        """
        Save the weights of the model in :class:`numpy.nrdarray` format.

        :param param_file:
            path to the weight file.
        """

        assert not hasattr(super(), 'save')
        params_np = utils.bulk_to_numpy(self.params)
        params_dict = OrderedDict(zip(list(self.state_dict().keys()), params_np))
        T.save(params_dict, param_file)
        print('Model weights dumped to %s' % param_file)

    def load(self, param_file, eval=True):
        """
        Load the `numpy.ndarray` weights from file.

        :param param_file:
            path to the weight file.
        :param eval:
            whether to use evaluation mode or not.
        """

        assert not hasattr(super(), 'load')
        params_dict = T.load(param_file)
        self.load_state_dict(params_dict)
        if eval:
            self.eval()
        print('Model weights loaded from %s' % param_file)

    def reset_parameters(self):
        """
        This overloads the :meth:`torch.Module.reset_parameters` of the module.
        Used for custom weight initialization.
        """

        assert not hasattr(super(), 'reset_parameters')
        pass


class Module(nn.Module, _LayerMethod):
    pass


class MultiSingleInputModule(Module):
    """
    This is an abstract class.
    This class computes the results of multiple modules given an input tensor,
    then fuses the results.

    Parameters
    ----------
    modules_or_tensors
        a list of modules or tensors whose results are fused together.

    Attributes
    ----------
    input_shape
        a list of input shapes of the incoming modules and tensors.
    """

    def __init__(self, *modules_or_tensors):
        assert all(isinstance(item, (nn.Module, T.Tensor)) for item in modules_or_tensors), \
            'All items in modules_or_tensors should be Pytorch modules or tensors'

        super().__init__()

        def foo(item):
            idx = len(list(self.children()))
            if isinstance(item, nn.Module):
                self.add_module(f'module{idx}', item)
            else:
                self.add_module(f'tensor{idx}', Lambda(lambda *args, **kwargs: item))

        list(map(foo, modules_or_tensors))

    def forward(self, input):
        outputs = [module(input) for name, module in self.named_children()]
        return tuple(outputs)

    @property
    def trainable(self):
        return tuple()

    @property
    def regularizable(self):
        return tuple()


class MultiMultiInputModule(MultiSingleInputModule):
    """
    Similar to :class:`MultiSingleInputModule`, but each module has its own input tensor.
    """

    def __init__(self, *modules_or_tensors):
        super().__init__(*modules_or_tensors)

    def forward(self, *input, **kwargs):
        input_it = iter(input)
        outputs = [module(next(input_it), **kwargs) if name.startswith('module') else module()
                   for name, module in self.named_children()]
        return tuple(outputs)


class SingleMultiInputModule(Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *input, **kwargs):
        return tuple([self.module(inp, **kwargs) for inp in input])

    @property
    def trainable(self):
        return tuple()

    @property
    def regularizable(self):
        return tuple()


class Sequential(nn.Sequential, _LayerMethod):
    """
    Similar to :class:`torch.nn.Sequential`, but extended by
    :class:`~neural_toolkits.layers.layers._LayerMethod`.
    All the usages in native Pytorch are preserved.

    Parameters
    ----------
    args
        a list of modules as in :class:`torch.nn.Sequential`.
    """

    def reset_parameters(self):
        for m in self.children():
            m.reset_parameters()


def wrapper(*args, **kwargs):
    """
    A class decorator to wrap any :mod:`torch` module.

    :param args:
        extra arguments needed by the module.
    :param kwargs:
        extra keyword arguments needed by the module.
    :return:
        The input module extended by :class:`~neural_toolkits.layers.layers._LayerMethod`.

    Examples
    --------
    You can use this function directly on any :mod:`torch` module

    >>> import torch.nn as nn
    >>> import neural_toolkits as ntk
    >>> dropout = ntk.wrapper(p=.2)(nn.Dropout2d)() # because wrapper returns a class!

    Alternatively, you can use it as a decorator

    .. code-block:: python

        import torch.nn as nn
        import neural_toolkits as ntk

        @ntk.wrapper(# optional arguments for input and output shapes)
        class Foo(nn.Module):
            ...

        foo = Foo()
    """

    def decorator(module: nn.Module):
        assert issubclass(module, nn.Module), 'module must be a subclass of Pytorch\'s Module'

        class _Wrapper(module, _LayerMethod):
            def __init__(self):
                super().__init__(*args, **kwargs)

            def forward(self, input, *args, **kwargs):
                return super().forward(input, *args, **kwargs)

        _Wrapper.__name__ = module.__name__
        _Wrapper.__doc__ = module.__doc__
        _Wrapper.__module__ = module.__module__
        return _Wrapper
    return decorator


@add_custom_repr
class Lambda(Module):
    """
    Wraps a function as a :class:`~neural_toolkits.layers.Module`.

    Parameters
    ----------
    func
        a callable function.
    input_shape
        shape of the input tensor.
    output_shape
        shape of the output tensor.
        If ``None``, the output shape is calculated by performing a forward pass.
    kwargs
        keyword arguments required by `func`.

    Examples
    --------
    You can easily wrap a :mod:`torch` function

    .. code-block:: python

        import torch as T
        import neural_toolkits as ntk

        a, b = T.rand(3, 1), T.rand(3, 2)
        cat = ntk.Lambda(T.cat, dim=1)
        c = cat((a, b))
        print(c.shape)

    Also, it works for any self-defined function as well

    .. code-block:: python

        import neural_toolkits as ntk

        def foo(x, y):
            return x + y

        a = T.rand(3, 3)
        print(a)
        foo_sum = ntk.Lambda(foo, y=1.)
        res = foo_sum(a)
        print(res)
    """

    def __init__(self, func, **kwargs):
        assert callable(func), 'The provided function must be callable'
        super().__init__()
        self.func = func
        self.kwargs = kwargs

    def forward(self, *input):
        return self.func(*input, **self.kwargs)

    def extra_repr(self):
        s = '{}'.format(self.func.__name__)
        return s


@add_custom_repr
class Eye(Sequential):
    def forward(self, input):
        return input
