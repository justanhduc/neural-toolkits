from collections.abc import Iterable
import torch.nn as nn
import torch as T

from . import root_logger

__all__ = ['deprecated', 'get_non_none', 'spectral_norm', 'remove_spectral_norm', 'add_custom_repr',
           'revert_sync_batchnorm']


class _BatchNormXd(T.nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        # The only difference between BatchNorm1d, BatchNorm2d, BatchNorm3d, etc
        # is this method that is overwritten by the sub-class
        # This original goal of this method was for tensor sanity checks
        # If you're ok bypassing those sanity checks (eg. if you trust your inference
        # to provide the right dimensional inputs), then you can just use this method
        # for easy conversion from SyncBatchNorm
        # (unfortunately, SyncBatchNorm does not store the original class - if it did
        #  we could return the one that was originally created)
        return


def add_custom_repr(cls):
    """
    A decorator to add a custom repr to the designated class.
    User should define extra_repr for the decorated class.

    :param cls:
        a subclass of :class:`~neural_toolkits.layers.Module`.
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


def remove_spectral_norm(module, name='weight', exclude=None):
    """
    Applies :func:`torch.nn.utils.remove_spectral_norm` recursively to `module` and all of
    its submodules.

    :param module:
        containing module.
    :param name:
        name of weight parameter.
        Default: ``'weight'``.
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
            module = nn.utils.remove_spectral_norm(module, name)

        return module
    else:
        for mod_name, mod in module.named_children():
            mod = remove_spectral_norm(mod, name, exclude=exclude)
            module.__setattr__(mod_name, mod)
        return module


def revert_sync_batchnorm(module):
    """
    From https://github.com/pytorch/pytorch/issues/41081#issuecomment-783961547.

    :param module:
        An instance of `torch.nn.Module`.
    :return:
        An instance of `torch.nn.Module` with `SyncBatchNorm` reverted to BatchNorm.
    """
    # this is very similar to the function that it is trying to revert:
    # https://github.com/pytorch/pytorch/blob/c8b3686a3e4ba63dc59e5dcfe5db3430df256833/torch/nn/modules/batchnorm.py#L679
    module_output = module
    if isinstance(module, T.nn.modules.batchnorm.SyncBatchNorm):
        module_output = _BatchNormXd(module.num_features,
                                     module.eps, module.momentum,
                                     module.affine,
                                     module.track_running_stats)
        if module.affine:
            with T.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig
    for name, child in module.named_children():
        module_output.add_module(name, revert_sync_batchnorm(child))
    del module
    return module_output
