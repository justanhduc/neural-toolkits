from collections.abc import Iterable
import torch.nn as nn

from . import root_logger

__all__ = ['deprecated', 'get_non_none', 'spectral_norm', 'add_custom_repr']


def add_custom_repr(cls):
    """
    A decorator to add a custom repr to the designated class.
    User should define extra_repr for the decorated class.

    :param cls:
        a subclass of :class:`~neuralnet_pytorch.layers.Module`.
    """

    def _repr(self):
        return self.__class__.__name__ + f'({self.extra_repr()})'

    setattr(cls, '__repr__', _repr)
    return cls


def deprecated(new_func, version):
    def _deprecated(func):
        """prints out a deprecation warning"""

        def func_wrapper(*args, **kwargs):
            root_logger.warning('%s is deprecated and will be removed in version %s. Use %s instead.' %
                                (func.__name__, version, new_func.__name__), exc_info=True)
            return func(*args, **kwargs)

        return func_wrapper
    return _deprecated


def get_non_none(array):
    """
    Gets the first item that is not ``None`` from the given array.

    :param array:
        an arbitrary array that is iterable.
    :return:
        the first item that is not ``None``.
    """
    assert isinstance(array, Iterable)

    try:
        e = next(item for item in array if item is not None)
    except StopIteration:
        e = None
    return e


def spectral_norm(module, name='weight', n_power_iterations=1, eps=1e-12, dim=None, exclude=None):
    """
    Applies :func:`torch.nn.utils.spectral_norm` recursively to `module` and all of
    its submodules.

    :param module:
        containing module.
    :param name:
        name of weight parameter.
        Default: ``'weight'``.
    :param n_power_iterations:
        number of power iterations to calculate spectral norm.
    :param eps:
        epsilon for numerical stability in calculating norms.
    :param dim:
        dimension corresponding to number of outputs,
        the default is ``0``, except for modules that are instances of
        ConvTranspose{1,2,3}d, when it is ``1``.
    :param exclude:
        a list of classes of which instances will not be applied SN.
    :return:
        the original module with the spectral norm hook.
    """
    excludes = [
        nn.modules.batchnorm._NormBase,
        nn.GroupNorm,
        nn.LayerNorm
    ]
    if exclude is not None:
        excludes = excludes + list(exclude)

    if hasattr(module, name):
        if not isinstance(module, tuple(excludes)):
            module = nn.utils.spectral_norm(module, name, n_power_iterations, eps, dim)

        return module
    else:
        for mod_name, mod in module.named_children():
            mod = spectral_norm(mod, name, n_power_iterations, eps, dim, exclude=exclude)
            module.__setattr__(mod_name, mod)
        return module
